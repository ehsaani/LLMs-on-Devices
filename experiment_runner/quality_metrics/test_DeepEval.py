import pandas as pd
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI

# ---------------------------
# 1) DATA SETUP
# ---------------------------
context_text = (
    "The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee "
    "in 1989. He was working at CERN, the European Organization for Nuclear "
    "Research, near Geneva, Switzerland. Berners-Lee created the Web to meet "
    "the demand for automatic information-sharing between scientists in "
    "universities and institutes around the world."
)

models_data = [
    {
        "index": 1,
        "model": "Qwen2-0.5B-Q4_K_M",
        "response": "Tim Berners-Lee invented the World Wide Web in 1989. He was working at CERN, a European organization for nuclear research, near Geneva, Switzerland. Berners-Lee created the web to meet the demand for automatic information-sharing between scientists in universities and institutes around the world.",
    },
    {
        "index": 2,
        "model": "Qwen2.5-1.5B-Q4_K_M",
        "response": "The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989. He created the Web to enable automatic information-sharing between scientists in universities and institutes around the world. Berners-Lee worked at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland.",
    },
    {
        "index": 3,
        "model": "Phi-2-2.8B-Q4_K_M",
        "response": "The World Wide Web (WWW) was created by British scientist Tim Berners-Lee in 1989 while working at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland. He developed the Web to facilitate the automatic sharing of information between scientists in different universities and research institutes worldwide.",
    },
    {
        "index": 4,
        "model": "Qwen2.5-3B-Q4_K_M",
        "response": "The World Wide Web (WWW) was invented in 1989 by Tim Berners-Lee, a British scientist working at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland. Berners-Lee developed the Web to facilitate automatic information-sharing among scientists globally.",
    },
    {
        "index": 5,
        "model": "OLMoE-6.9B-Q4_K_M",
        "response": "The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while he was working at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland. Berners-Lee created the Web to enable automatic information-sharing between scientists at universities and institutes worldwide, fulfilling a demand for such a system at CERN.",
    },
    {
        "index": 6,
        "model": "Qwen2.5-7.6B-Q4_K_M",
        "response": "The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while working at CERN. He developed the Web to facilitate automatic information-sharing among scientists worldwide.",
    },
    {
        "index": 7,
        "model": "Llama3.1-8B-Q4_K_M",
        "response": "The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while he was working at CERN in Switzerland. He created the Web to facilitate automatic information-sharing between scientists globally.",
    },
    {
        "index": 8,
        "model": "Gemma2-9B-Q4_K_M",
        "response": "Tim Berners-Lee, a British scientist working at CERN, invented the World Wide Web (WWW) in 1989 to facilitate easy information sharing among scientists globally.",
    },
    {
        "index": 9,
        "model": "Qwen2-0.5B-IQ4_XS",
        "response": "Tim Berners-Lee invented the World Wide Web in 1989. He was working at CERN, a European organization for nuclear research, near Geneva, Switzerland. The World Wide Web was created to meet the need for automatic information sharing between scientists in universities and institutes around the world.",
    },
    {
        "index": 10,
        "model": "Qwen2.5-1.5B-IQ4_XS",
        "response": "The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while working at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland. Berners-Lee created the Web to facilitate automatic information-sharing among scientists in universities and institutes worldwide.",
    },
    {
        "index": 11,
        "model": "Phi-2-2.8B-IQ4_XS",
        "response": "The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while working at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland. He created the Web to facilitate automatic information-sharing between scientists in universities and institutes around the world.",
    },
    {
        "index": 12,
        "model": "Qwen2.5-3B-IQ4_XS",
        "response": "The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while he was working at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland. He developed the Web to facilitate automatic information-sharing among scientists in various universities and research institutes worldwide.",
    },
    {
        "index": 13,
        "model": "OLMoE-6.9B-IQ4_XS",
        "response": "The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while he was working at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland. Berners-Lee created the Web to enable automatic information-sharing among scientists at universities and research institutions worldwide, fulfilling a demand for such a system at the time.",
    },
    {
        "index": 14,
        "model": "Qwen2.5-7.6B-IQ4_XS",
        "response": "The World Wide Web was invented by British scientist Tim Berners-Lee in 1989 while he was working at CERN in Switzerland. Berners-Lee developed the Web to facilitate automatic information sharing among scientists in universities and research institutions worldwide.",
    },
    {
        "index": 15,
        "model": "Llama3.1-8B-IQ4_XS",
        "response": "The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while he was working at CERN in Switzerland. He created the Web to facilitate automatic information-sharing between scientists worldwide.",
    },
    {
        "index": 16,
        "model": "Gemma2-9B-IQ4_XS",
        "response": "Tim Berners-Lee, a British scientist working at CERN, invented the World Wide Web (WWW) in 1989 to facilitate automatic information sharing among scientists globally.",
    },
]

# ---------------------------
# 2) METRIC DEFINITIONS
# ---------------------------
metrics_library = {

    "Faithfulness": [
        "Extract all specific claims, dates, and entities made in the 'Actual Output'.",
        "Cross-reference each extracted claim against the 'Input' (Source Text).",
        "CRITICAL: Identify any information in the output that contradicts the source or appears to be hallucinated (unsupported).",
        "High score ONLY if every fact is verifiable in the source text.",
        "Low score if the model invents major facts not present in the source.",
        "Output an overall model faithfulness score between 0.00 and 1.00."
    ],
    "Relevance": [ 
        "Analyze the 'Input' text and list the core information units (key entities like 'Tim Berners-Lee', actions, dates like '1989').",
        "Check the 'Actual Output' to see which of these core information units are present.",
        "Identify any critical concepts from the source that were omitted in the summary.",
        "High score if the summary provides a comprehensive overview of the source text's main message.",
        "Low score if the summary misses the main topic entirely.",
        "Output an overall model relevancy score between 0.00 and 1.00."
    ],
    "Coherence": [
        "Read the 'Actual Output' and analyze the logical progression of ideas.",
        "Check for smooth transitions between sentences and clauses (e.g., proper use of connectives).",
        "Scan specifically for 'Repetition Loops' (repeating the same phrase) or cut-off sentences.",
        "Ensure pronouns and references (e.g., 'he', 'it') clearly point to the correct entities.",
        "High score for perfect, natural English flow.",
        "Low score for disjointed lists, word salad, or severe grammatical errors.",
        "Output an overall model coherence score between 0.00 and 1.00."
    ],
    "Overall_Quality": [
        "Holistically evaluate the summary's utility.",
        "IMMEDIATE FAIL: If the response is a copy-paste of the source (or >80 percent similarity), the MAXIMUM score is 0.20.",
        "Check for 'lazy' summarization: Does the model just delete a few adjectives but keep the exact sentence structure?",
        "Look for 'synthesis': Did the model combine multiple source sentences into one new, efficient sentence?",
        "Reward models that use their own vocabulary to convey the full meaning in significantly fewer words.",
        "High score for excellent, high-compression summarization.",
        "Output an overall model quality score between 0.00 and 1.00."
    ]
}

# ---------------------------
# 3) STRUCTURED OUTPUT DEFINITION
# ---------------------------
class ModelScore(BaseModel):
    # Changed from model_name (str) to model_id (int) to prevent hallucinations
    model_id: int 
    score: float = Field(..., description="Score from 0.00 to 1.00")

class MetricVerdict(BaseModel):
    rankings: List[ModelScore]

# ---------------------------
# 4) THE RELATIVE SCORING ENGINE
# ---------------------------
def evaluate_specific_metric(metric_name, steps, context, models):
    client = OpenAI()
    
    # Map IDs to Names for safe retrieval later
    id_map = {m['index']: m['model'] for m in models}

    # Format the candidate text using IDs only
    candidates_text = ""
    for m in models:
        candidates_text += f"--- MODEL ID: {m['index']} ---\n{m['response']}\n\n"

    steps_text = "\n".join([f"- {s}" for s in steps])

    system_prompt = (
        f"You are an expert evaluator grading AI summaries on the metric: {metric_name.upper()}.\n\n"
        f"EVALUATION STEPS:\n{steps_text}\n\n"
        "INSTRUCTIONS:\n"
        "1. Read the SOURCE CONTEXT.\n"
        "2. Read all MODEL CANDIDATES (identified by ID).\n"
        "3. Compare the candidates AGAINST EACH OTHER based on the Evaluation Steps.\n"
        "4. Assign a score (0.00 - 1.00). The best model relative to the others should get the highest score.\n"
        "5. Be strict. If a model hallucinates, give it a low score on Consistency."
    )

    user_prompt = f"SOURCE CONTEXT:\n{context}\n\nCANDIDATES:\n{candidates_text}"

    # Updated to use standard beta.chat.completions.parse
    response = client.beta.chat.completions.parse(
        model="gpt-5.2", # Ensure a structured-output capable model is used
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        seed=42, # Forces deterministic output
        response_format=MetricVerdict,
    )   
    
    rankings = response.choices[0].message.parsed.rankings
    
    # Map IDs back to real names
    results_mapped = []
    for r in rankings:
        results_mapped.append({
            "model_name": id_map.get(r.model_id, f"Unknown_ID_{r.model_id}"),
            "score": r.score
        })
        
    return results_mapped

# ---------------------------
# 5) MAIN EXECUTION LOOP
# ---------------------------
if __name__ == "__main__":
    
    final_data = []

    # Loop through every metric defined
    for metric_name, steps in metrics_library.items():
        print(f"📊 Evaluating metric: {metric_name}...")
        
        # Run the relative comparison for this metric
        results = evaluate_specific_metric(metric_name, steps, context_text, models_data)
        
        # Store results
        for res in results:
            final_data.append({
                "Model": res["model_name"],
                "Metric": metric_name,
                "Score": res["score"],
            })

    # ---------------------------
    # 6) REPORTING
    # ---------------------------
    df = pd.DataFrame(final_data)
    
    # Pivot table: Models as Rows, Metrics as Columns
    pivot_df = df.pivot(index="Model", columns="Metric", values="Score")
    
    # Reset index to make 'Model' a regular column again
    pivot_df = pivot_df.reset_index()
    
    # Remove the index name for cleaner output
    pivot_df.columns.name = None
    
    print("\n" + "="*60)
    print("🏆 FINAL RELATIVE SCORECARD (0 - 1)")
    print("="*60)
    print(pivot_df)
    
    # Save full details to CSV
    csv_file = "llm_a_judge_scores.csv"
    pivot_df.to_csv(csv_file, index=False)
    print(f"\n✅ Detailed results saved to {csv_file}")