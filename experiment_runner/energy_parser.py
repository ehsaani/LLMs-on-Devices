import re
import json
import os

file_path = "llama_output.txt"

def parse_llama_log_file(file_path):
    """
    Parses the llama output log file using Regex to extract timing metrics
    and JSON parsing to extract the final clean response.
    """
    metrics = {
        'model_response': '',
        'input_token_count': 0,
        'output_token_count': 0,
        'total_token_count': 0,
        'prompt_prefill_speed': 0.0,
        'generation_decoder_speed': 0.0,
        'prefill_latency': 0.0,
        'generation_latency': 0.0,
        'inference_latency': 0.0,
        'time_to_first_token': 0.0
    }

    # Regex Patterns
    # 1. Prompt Eval: Matches "prompt eval time = ..."
    prompt_pattern = re.compile(r"prompt eval time\s+=\s+(\d+\.\d+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+(\d+\.\d+)\s+ms per token,\s+(\d+\.\d+)\s+tokens per second\)")
    
    # 2. Eval (Generation): Uses (?<!prompt) to ensure matches do not include "prompt eval time"
    #    Matches "eval time = ..." but NOT "prompt eval time = ..."
    eval_pattern = re.compile(r"(?<!prompt)\s+eval time\s+=\s+(\d+\.\d+)\s+ms\s+/\s+(\d+)\s+(?:tokens|runs).*?\(\s+(\d+\.\d+)\s+ms per token,\s+(\d+\.\d+)\s+tokens per second\)")
    
    # 3. Total Time
    total_pattern = re.compile(r"total time\s+=\s+(\d+\.\d+)\s+ms")
    
    # 4. Response JSON
    response_pattern = re.compile(r'Parsed message: (\{.*\})')

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return metrics

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

            # 1. Extract Prompt Metrics
            prompt_match = prompt_pattern.search(content)
            if prompt_match:
                metrics['prefill_latency'] = float(prompt_match.group(1)) / 1000.0  # ms -> s
                metrics['input_token_count'] = int(prompt_match.group(2))
                metrics['prompt_prefill_speed'] = float(prompt_match.group(4))

            # 2. Extract Generation Metrics & TTFT
            eval_match = eval_pattern.search(content)
            if eval_match:
                metrics['generation_latency'] = float(eval_match.group(1)) / 1000.0  # ms -> s
                metrics['output_token_count'] = int(eval_match.group(2))
                ms_per_token = float(eval_match.group(3))
                metrics['generation_decoder_speed'] = float(eval_match.group(4))
                
                # Calculate TTFT: Prefill time + time for 1 decode step
                metrics['time_to_first_token'] = metrics['prefill_latency'] + (ms_per_token / 1000.0)

            # 3. Extract Total Metrics
            total_match = total_pattern.search(content)
            if total_match:
                metrics['inference_latency'] = float(total_match.group(1)) / 1000.0  # ms -> s
            else:
                # Fallback if total time line is missing
                metrics['inference_latency'] = metrics['prefill_latency'] + metrics['generation_latency']

            metrics['total_token_count'] = metrics['input_token_count'] + metrics['output_token_count']
            
            # 4. Extract Model Response (JSON Method)
            response_match = response_pattern.search(content)
            if response_match:
                json_str = response_match.group(1)
                try:
                    data = json.loads(json_str)
                    metrics['model_response'] = data.get('content', '')
                except json.JSONDecodeError:
                    metrics['model_response'] = "Error parsing JSON response content"
            else:
                # Fallback to simple cleanup if JSON line not found
                # Replaced missing 'self._fallback_clean_response' with direct logic
                metrics['model_response'] = _fallback_clean_response(content)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return metrics

def _fallback_clean_response(content):
    """
    Simple helper to extract text if JSON parsing fails.
    """
    # Simple logic: return the last 500 chars if specific parsing isn't possible
    return "Raw content extraction (JSON pattern not found)"

# Execution
if __name__ == "__main__":
    # Ensure a dummy file exists for testing if needed, or rely on existing file
    result = parse_llama_log_file(file_path)
    print(json.dumps(result, indent=4))