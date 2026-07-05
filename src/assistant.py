import sys
import io
import json
import ollama
import time
from pydantic import BaseModel, Field
from typing import Optional

# ==========================================
# 1. DEFINE THE STRICT STRUCTURE CONTRACT
# ==========================================
class AssistantResponse(BaseModel):
    thought: str = Field(..., description="Deep step-by-step reasoning process explaining the solution strategy.")
    action: Optional[str] = Field(None, description="Must be exactly 'execute_python_code' if computation/math is needed, otherwise null.")
    code_payload: Optional[str] = Field(None, description="The raw, valid Python script containing print statements to extract data.")
    reply: str = Field(..., description="The highly accurate conversational answer formatted for the end-user.")

# ==========================================
# 2. ISOLATED PYTHON SANDBOX TOOL
# ==========================================
def run_python_sandbox(code: str) -> str:
    """Executes a text script block safely by capturing standard output streams."""
    output_buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = output_buffer
    
    try:
        # Create a single shared dictionary for both globals and locals
        unified_scope = {}
        
        # CRITICAL FIX: Passing unified_scope as BOTH the global and local argument
        # tells exec() to treat the script exactly like a standard file execution context.
        exec(code, unified_scope, unified_scope)
        
        sys.stdout = old_stdout
        captured_output = output_buffer.getvalue()
        return captured_output if captured_output.strip() else "Code ran successfully, but returned no print output."
    except Exception as error:
        sys.stdout = old_stdout
        return f"Runtime Sandbox Error: {str(error)}"

# ==========================================
# 3. MAIN ORCHESTRATION LAYER
# ==========================================
def start_local_engine():
    print("\n🤖 Local Agentic AI Assistant Online [METRICS ENABLED].")
    print("Path: /Users/hemachandiran/Projects/offline-llm-benchmark/src\n")
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert engineering assistant with execution capabilities.\n"
                "CRITICAL: If a user query requires ANY math, calculation, or loops, you MUST use code. "
                "Set 'action' to 'execute_python_code' and place the full executable script inside 'code_payload'."
            )
        }
    ]

    while True:
        try:
            user_prompt = input("You 👤: ")
            if user_prompt.lower() in ['exit', 'quit']:
                break
            if not user_prompt.strip():
                continue

            messages.append({"role": "user", "content": user_prompt})
            
            # --- START METRICS CAPTURE ---
            start_time = time.perf_counter()
            
            raw_response = ollama.chat(
                model="qwen3.5:9b",
                messages=messages,
                format=AssistantResponse.model_json_schema(),
                options={"temperature": 0.0}
            )
            
            end_time = time.perf_counter()
            # ------------------------------
            
            # Extract payload analytics
            content_string = raw_response.message.content
            total_duration = end_time - start_time
            
            # Approximate token counts (standard word-to-token ratio estimation)
            estimated_tokens = len(content_string.split()) * 1.3
            tokens_per_second = estimated_tokens / total_duration if total_duration > 0 else 0
            
            MAX_RETRIES = 3

for attempt in range(MAX_RETRIES):
    raw = ollama.chat(
        model="qwen3.5:9b",
        messages=messages,
        format=AssistantResponse.model_json_schema(),
        options={"temperature": 0.0}
    )
    
    try:
        parsed = AssistantResponse.model_validate_json(raw.message.content)
        break  # Success! Exit retry loop
    except Exception as e:
        error_msg = str(e)
        print(f"  ⚠️ Attempt {attempt + 1} failed: {error_msg[:60]}...")
        
        if attempt < MAX_RETRIES - 1:
            # Tell the model what went wrong and ask it to fix
            messages.append({
                "role": "assistant",
                "content": raw.message.content
            })
            messages.append({
                "role": "user",
                "content": f"Fix this JSON error: {error_msg}. Output ONLY valid JSON matching the schema."
            })
        else:
            # Last resort: use a default fallback
            parsed = AssistantResponse(
                thought="Failed to generate valid response after retries.",
                action=None,
                code_payload=None,
                reply="I couldn't generate a valid response. Please try rephrasing."
            )

        except Exception as runtime_error:
            print(f"❌ Structural Parsing Failure: {runtime_error}\n")

if __name__ == "__main__":
    start_local_engine()