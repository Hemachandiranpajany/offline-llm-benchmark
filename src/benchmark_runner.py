import time
import json
import csv
import os
import ollama
import psutil
import subprocess
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
from report_generator import generate_report

# Reuse the exact same response schema contract
class BenchmarkResponse(BaseModel):
    thought: str = Field(..., description="Reasoning.")
    action: Optional[str] = Field(None, description="Action.")
    code_payload: Optional[str] = Field(None, description="Payload.")
    reply: str = Field(..., description="Reply.")

# 1. Define models to test and standard cross-disciplinary prompts
MODELS_TO_TEST = ["llama3.2:3b", "qwen3.5:9b", "gemma4:e4b"]

TEST_PROMPTS = [
    {"category": "Logic / Math", "prompt": "Calculate the 15th Fibonacci number, multiply it by 12, and output the result."},
    {"category": "Data Extraction", "prompt": "Extract names and roles from this text into a clean list: 'Alice is the lead architect, Bob is DevOps, and Charlie handles QA.'"},
    {"category": "Code Generation", "prompt": "Write a python function to validate an IPv4 address string without using external libraries."}
]

def execute_single_test(model_name: str, prompt_text: str):
    """Executes a structured turn and yields raw performance timings."""
    messages = [{"role": "user", "content": prompt_text}]
    
    start = time.perf_counter()
    response = ollama.chat(
        model=model_name,
        messages=messages,
        format=BenchmarkResponse.model_json_schema(),
        options={"temperature": 0.0}
    )
    end = time.perf_counter()

    content_string = response.message.content
    duration = end - start

    MAX_RETRIES = 3
    parsed = None

    for attempt in range(MAX_RETRIES):
        try:
            parsed = BenchmarkResponse.model_validate_json(content_string)
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return {"status": "FAIL: Validation failed after retries", "time": 0, "tps": 0, "tokens": 0, "memory_mb": 0}

            messages.append({"role": "assistant", "content": content_string})
            messages.append({"role": "user", "content": f"Fix this JSON error: {str(e)}. Output ONLY valid JSON matching the schema."})

            retry_response = ollama.chat(
                model=model_name,
                messages=messages,
                format=BenchmarkResponse.model_json_schema(),
                options={"temperature": 0.0}
            )
            content_string = retry_response.message.content

    char_count = len(content_string)
    estimated_tokens = char_count / 4.0
    tps = estimated_tokens / duration if duration > 0 else 0

    process = psutil.Process()
    mem_mb = process.memory_info().rss / (1024 * 1024)

    return {"status": "PASS", "time": duration, "tps": tps, "tokens": int(estimated_tokens), "memory_mb": round(mem_mb, 1)}

def run_suite():
    print("🚀 Starting Offline LLM Agent Benchmark Pipeline...")
    results = []

    for model in MODELS_TO_TEST:
        print(f"\n🔄 Activating Local Model Weight Layer: [{model}]")
        print("  🔥 Warming up...", end="", flush=True)
        execute_single_test(model, "List the numbers 1 to 20 separated by commas. No other text.")
        print(" done")
        for item in TEST_PROMPTS:
            print(f"  📝 Testing Category: {item['category']}... ", end="", flush=True)
            metrics = execute_single_test(model, item['prompt'])
            print(f"[{metrics['status']}] ({metrics['time']:.2f}s | {metrics['tps']:.1f} TPS)")
            
            results.append({
                "model": model,
                "category": item['category'],
                "duration": metrics['time'],
                "tps": metrics['tps'],
                "tokens": metrics['tokens'],
                "memory_mb": metrics['memory_mb'],
                "status": metrics['status'],
            })

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(results_dir, f"benchmark_report_{timestamp}.md")
    report_content = generate_report(results)
    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"\n📊 Performance testing execution finished completely! Results saved to: {report_path}")

if __name__ == "__main__":
    run_suite()