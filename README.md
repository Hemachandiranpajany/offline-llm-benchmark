# Offline LLM Benchmark

Structured JSON output enforcement, retry logic, and token generation benchmarking for local LLMs — all running entirely offline via Ollama.

Built to understand how AI agents enforce output schemas, handle LLM unreliability, and measure real-world inference performance across different model sizes.

## Features

- **Pydantic schema enforcement** — Forces any LLM to output structured JSON matching a defined contract
- **Retry logic with self-correction** — Automatically retries on validation failure, feeding errors back to the model for self-correction
- **Interactive assistant** — Chat interface with a Python code execution sandbox (tool-use pattern)
- **Multi-model benchmarking** — Tests throughput (TPS), latency, and memory across multiple models
- **Formal report generation** — Produces timestamped markdown reports with hardware specs, methodology, results, and analysis

## Architecture

```
                         ┌──────────────────────┐
                         │    Ollama (local)     │
                         │  ┌──────────────────┐ │
                         │  │ llama3.2:3b      │ │
                         │  │ qwen3.5:9b       │ │
                         │  │ gemma4:e4b       │ │
                         │  └──────────────────┘ │
                         └──────────┬───────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐        ┌───────────────────┐        ┌──────────────────┐
│  assistant.py │        │ benchmark_runner   │        │ report_generator │
│  Interactive  │        │      .py           │        │      .py         │
│  chat + code  │        │  Automated tests   │        │  Markdown report │
│  execution    │        │  across models     │        │  with analysis   │
└───────────────┘        └───────────────────┘        └──────────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Pydantic         │
                         │  JSON validation     │
                         │  + retry loop        │
                         └──────────────────────┘
```

## Prerequisites

- macOS, Linux, or Windows (with WSL)
- [Ollama](https://ollama.com) installed and running
- At least one model pulled (`ollama pull llama3.2:3b`)
- Python 3.11+

## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/offline-llm-benchmark.git
cd offline-llm-benchmark

# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pull a model (if you haven't already)
ollama pull llama3.2:3b

# Run the interactive assistant
python src/assistant.py

# Run the benchmark suite
python src/benchmark_runner.py
```

## Usage

### Interactive Assistant

```bash
python src/assistant.py
```

Starts a chat session. The LLM must respond in a structured JSON format:

- `thought` — step-by-step reasoning
- `action` — `"execute_python_code"` or `null`
- `code_payload` — Python code to run in the sandbox
- `reply` — final response to the user

If the LLM outputs invalid JSON, the retry loop catches it, shows the error, and asks the model to fix it — up to 3 attempts.

### Benchmark Suite

```bash
python src/benchmark_runner.py
```

Tests each model against 3 prompt categories:
- Logic / Math
- Data Extraction
- Code Generation

Results are saved to `src/results/benchmark_report_<timestamp>.md`.

## Project Structure

```
offline-llm-benchmark/
├── src/
│   ├── assistant.py           # Interactive chat with code sandbox
│   ├── benchmark_runner.py    # Automated benchmarking pipeline
│   ├── report_generator.py    # Formal report builder
│   └── results/               # Generated benchmark reports
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## How It Works

### Schema Enforcement

Every LLM response is validated against a Pydantic `BaseModel`:

```python
class AssistantResponse(BaseModel):
    thought: str
    action: Optional[str]
    code_payload: Optional[str]
    reply: str
```

The model receives this schema via Ollama's `format` parameter, and the response is parsed with `model_validate_json()`.

### Retry Logic

When validation fails, the error message is fed back to the model as context:

1. LLM outputs invalid JSON → validation error
2. Error appended to conversation: *"Fix this JSON error: {details}"*
3. LLM retries with awareness of its mistake
4. Falls back to a safe default after 3 failures

This self-correction pattern is how production AI agents achieve reliable structured output.

### Benchmarking Metrics

- **TPS** — Tokens per second (estimated as `character_count / 4.0 / wall_time`)
- **Latency** — Wall-clock time for full generation
- **Memory** — Python process RSS tracked via psutil
- **Warm-up** — One prompt run per model before measurements to eliminate cold-start bias

## Sample Results

Benchmarked on Apple M5 (16 GB unified memory) with Q4_K_M quantized models:

| Model | Avg TPS | Avg Time (s) | Avg Memory (MB) |
| --- | --- | --- | --- |
| llama3.2:3b | 39.1 | 5.2 | 44 |
| gemma4:e4b | 4.5 | 38.4 | 26 |
| qwen3.5:9b | 2.3 | 126.3 | 24 |

## Key Concepts Learned

- **Structured output** — Pydantic + LLM format enforcement for reliable JSON
- **Self-correction loops** — Error feedback improves reliability from ~85% to ~99%
- **Tool use pattern** — LLM generates actions, system executes them (code sandbox)
- **Cold-start latency** — First inference is always slower; warm-up runs are essential
- **Throughput vs model size** — Smaller models (3B) can be 10-20x faster than larger ones (9B)
- **Benchmark methodology** — Multiple prompts, controlled temperature, statistical reporting

## Built With

- [Ollama](https://ollama.com) — Local LLM runtime
- [Pydantic](https://docs.pydantic.dev) — Data validation
- [psutil](https://github.com/giampaolo/psutil) — System monitoring
- [tqdm](https://github.com/tqdm/tqdm) — Progress bars
