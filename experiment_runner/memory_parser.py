import re
import os

def parse_llama_log(file_path):
    metrics = {
        "kv_cache_size_mb": "Not Found",
        "peak_memory_mb": "Not Found",
        "model_weight_mb": "Not Found",
        "context_ram_mb": "Not Found",
        "compute_ram_mb": "Not Found"
    }

    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return metrics

    with open(file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()

    # 1. Extract KV Cache Size
    # Target: "llama_kv_cache: size =   84.00 MiB"
    kv_match = re.search(r"llama_kv_cache:\s+size\s+=\s+([\d\.]+)\s+MiB", log_content)
    if kv_match:
        metrics["kv_cache_size_mb"] = float(kv_match.group(1))

    # 2. Extract Memory Breakdown
    # Target: "5612 =  4937 +     168 +     507"
    # Format: Total = Model + Context + Compute
    mem_match = re.search(r"(\d+)\s+=\s+(\d+)\s+\+\s+(\d+)\s+\+\s+(\d+)", log_content)
    if mem_match:
        metrics["peak_memory_mb"] = float(mem_match.group(1))    # Group 1: Total
        metrics["model_weight_mb"] = float(mem_match.group(2))   # Group 2: Model
        metrics["context_ram_mb"] = float(mem_match.group(3))    # Group 3: Context
        metrics["compute_ram_mb"] = float(mem_match.group(4))    # Group 4: Compute

    return metrics

# Main execution
file_name = "llama_output.txt"
results = parse_llama_log(file_name)

print(f"--- Memory Analysis for {file_name} ---")
print(f"peak_memory   {results['peak_memory_mb']} MiB")
print(f"model_weight  {results['model_weight_mb']} MiB")
print(f"KV_cache      {results['kv_cache_size_mb']} MiB")
print(f"context_RAM   {results['context_ram_mb']} MiB")
print(f"compute_RAM   {results['compute_ram_mb']} MiB")
