from datetime import datetime
import subprocess
import platform
from collections import defaultdict


def collect_system_info():
    uname = platform.uname()
    ollama_version = subprocess.run(
        ["ollama", "--version"], capture_output=True, text=True
    ).stdout.strip()

    model_details = {}
    try:
        models_out = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True
        ).stdout
        model_details["raw_list"] = models_out
    except Exception:
        model_details["raw_list"] = ""

    return {
        "system": f"{uname.system} {uname.machine}",
        "node": uname.node,
        "processor": platform.processor(),
        "python": platform.python_version(),
        "ollama_version": ollama_version,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def compute_stats(results):
    stats = defaultdict(list)
    for r in results:
        stats[r["model"]].append(r)

    summary = {}
    for model, runs in stats.items():
        tps_list = [r["tps"] for r in runs]
        time_list = [r["duration"] for r in runs]
        mem_list = [r["memory_mb"] for r in runs]
        summary[model] = {
            "avg_tps": sum(tps_list) / len(tps_list),
            "avg_time": sum(time_list) / len(time_list),
            "min_tps": min(tps_list),
            "max_tps": max(tps_list),
            "avg_memory_mb": sum(mem_list) / len(mem_list),
            "total_time": sum(time_list),
        }
    return summary


def generate_report(results):
    if not results:
        return "# Offline LLM Benchmark Report\n\n*No results to report.*\n"

    info = collect_system_info()
    summary = compute_stats(results)

    fastest_model = max(summary, key=lambda m: summary[m]["avg_tps"])
    most_efficient = min(summary, key=lambda m: summary[m]["avg_memory_mb"])

    lines = []
    lines.append("# Offline LLM Benchmark Report\n")
    lines.append(f"**Date:** {info['timestamp']}\n")
    lines.append("---\n")
    lines.append("## 1. Hardware & Software Configuration\n")
    lines.append("| Component | Detail |")
    lines.append("| --- | --- |")
    lines.append(f"| System | {info['system']} |")
    lines.append(f"| Processor | {info['processor']} |")
    lines.append(f"| Ollama Version | {info['ollama_version']} |")
    lines.append(f"| Python | {info['python']} |")
    lines.append("")
    lines.append("## 2. Methodology\n")
    lines.append("- **Token estimation:** Character count / 4.0 (1 token approx 4 chars)")
    lines.append("- **TPS:** Tokens generated / wall clock time in seconds")
    lines.append("- **Validation:** Pydantic schema enforced with 3 retry attempts")
    lines.append("- **Temperature:** 0.0 (deterministic output)")
    lines.append("- **Memory:** Python process RSS via psutil (not model VRAM)")
    lines.append("- **Warm-up:** One short prompt per model before measurements to load model into memory")
    lines.append("")
    lines.append("## 3. Raw Results\n")
    lines.append("| Model | Category | Time (s) | TPS | Tokens | Memory (MB) | Status |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        status_display = "OK" if r["status"] == "PASS" else f"FAIL"
        lines.append(
            f"| {r['model']} | {r['category']} | {r['duration']:.2f} | "
            f"{r['tps']:.1f} | {r['tokens']} | {r['memory_mb']} | {status_display} |"
        )
    lines.append("")
    lines.append("## 4. Per-Model Summary\n")
    lines.append("| Model | Avg TPS | Min TPS | Max TPS | Avg Time (s) | Avg Memory (MB) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for model, s in summary.items():
        lines.append(
            f"| {model} | {s['avg_tps']:.1f} | {s['min_tps']:.1f} | "
            f"{s['max_tps']:.1f} | {s['avg_time']:.1f} | {s['avg_memory_mb']:.0f} |"
        )
    lines.append("")
    lines.append("## 5. Analysis\n")
    lines.append(f"- **Fastest model:** {fastest_model} ({summary[fastest_model]['avg_tps']:.1f} avg TPS)")
    lines.append(f"- **Most memory-efficient:** {most_efficient} ({summary[most_efficient]['avg_memory_mb']:.0f} MB avg)")
    for model, s in summary.items():
        lines.append(f"- **{model}:** {s['avg_tps']:.1f} avg TPS, {s['avg_memory_mb']:.0f} MB, {s['total_time']:.1f}s total")
    lines.append("")
    lines.append("## 6. Recommendations\n")
    lines.append("- Add a warm-up run before each model to eliminate cold-start bias")
    lines.append("- Run each prompt 3+ times for statistical significance")
    lines.append("- Use `ollama ps` for real GPU memory instead of Python process RSS")
    lines.append("- Consider 4+ retries for models prone to JSON syntax errors")
    lines.append("")

    return "\n".join(lines)
