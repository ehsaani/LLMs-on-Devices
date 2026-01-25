import pandas as pd
import io
import textwrap
from bert_score import score

def main():
    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    # Use dedent to remove leading whitespace from the triple-quoted string
    csv_data = textwrap.dedent("""
                               index,model,response
                               1,Qwen2-0.5B-Q4_K_M,"Tim Berners-Lee invented the World Wide Web in 1989. He was working at CERN, a European organization for nuclear research, near Geneva, Switzerland. Berners-Lee created the web to meet the demand for automatic information-sharing between scientists in universities and institutes around the world."
                               2,Qwen2.5-1.5B-Q4_K_M,"The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989. He created the Web to enable automatic information-sharing between scientists in universities and institutes around the world. Berners-Lee worked at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland."
                               3,Phi-2-2.8B-Q4_K_M,"The World Wide Web (WWW) was created by British scientist Tim Berners-Lee in 1989 while working at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland. He developed the Web to facilitate the automatic sharing of information between scientists in different universities and research institutes worldwide."
                               4,Qwen2.5-3B-Q4_K_M,"The World Wide Web (WWW) was invented in 1989 by Tim Berners-Lee, a British scientist working at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland. Berners-Lee developed the Web to facilitate automatic information-sharing among scientists globally."
                               5,OLMoE-6.9B-Q4_K_M,"The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while he was working at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland. Berners-Lee created the Web to enable automatic information-sharing between scientists at universities and institutes worldwide, fulfilling a demand for such a system at CERN."
                               6,Qwen2.5-7.6B-Q4_K_M,"The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while working at CERN. He developed the Web to facilitate automatic information-sharing among scientists worldwide."
                               7,Llama3.1-8B-Q4_K_M,"The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while he was working at CERN in Switzerland. He created the Web to facilitate automatic information-sharing between scientists globally."
                               8,Gemma2-9B-Q4_K_M,"Tim Berners-Lee, a British scientist working at CERN, invented the World Wide Web (WWW) in 1989 to facilitate easy information sharing among scientists globally."
                               9,Qwen2-0.5B-IQ4_XS,"Tim Berners-Lee invented the World Wide Web in 1989. He was working at CERN, a European organization for nuclear research, near Geneva, Switzerland. The World Wide Web was created to meet the need for automatic information sharing between scientists in universities and institutes around the world."
                               10,Qwen2.5-1.5B-IQ4_XS,"The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while working at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland. Berners-Lee created the Web to facilitate automatic information-sharing among scientists in universities and institutes worldwide."
                               11,Phi-2-2.8B-IQ4_XS,"The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while working at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland. He created the Web to facilitate automatic information-sharing between scientists in universities and institutes around the world."
                               12,Qwen2.5-3B-IQ4_XS,"The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while he was working at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland. He developed the Web to facilitate automatic information-sharing among scientists in various universities and research institutes worldwide."
                               13,OLMoE-6.9B-IQ4_XS,"The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while he was working at CERN, the European Organization for Nuclear Research, near Geneva, Switzerland. Berners-Lee created the Web to enable automatic information-sharing among scientists at universities and research institutions worldwide, fulfilling a demand for such a system at the time."
                               14,Qwen2.5-7.6B-IQ4_XS,"The World Wide Web was invented by British scientist Tim Berners-Lee in 1989 while he was working at CERN in Switzerland. Berners-Lee developed the Web to facilitate automatic information sharing among scientists in universities and research institutions worldwide."
                               15,Llama3.1-8B-IQ4_XS,"The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee in 1989 while he was working at CERN in Switzerland. He created the Web to facilitate automatic information-sharing between scientists worldwide."
                               16,Gemma2-9B-IQ4_XS,"Tim Berners-Lee, a British scientist working at CERN, invented the World Wide Web (WWW) in 1989 to facilitate automatic information sharing among scientists globally."
                               """).strip()
  
    df = pd.read_csv(io.StringIO(csv_data))

    gold_summary = (
        "The World Wide Web was created by Tim Berners-Lee in 1989 at CERN to enable efficient information sharing among researchers, "
        "later evolving into a global information platform."
    )

    candidates = df["response"].tolist()
    references = [gold_summary] * len(candidates)

    # ==========================================
    # 3. BERTSCORE (Windows-safe)
    # ==========================================
    print("Calculating BERTScore...")
    
    # Removed idf=True because calculating IDF on a single repeated reference 
    # or small batch is statistically insignificant.
    P, R, F1 = score(
        candidates,
        references,
        lang="en",
        model_type="roberta-large",
        rescale_with_baseline=False,
        nthreads=1,
        verbose=True,
    )

    df["BERT_Precision"] = P.tolist()
    df["BERT_Recall"] = R.tolist()
    df["BERT_F1"] = F1.tolist()

    # Define the quality metric to avoid KeyError during sorting
    df["Final_Summary_Quality"] = df["BERT_F1"]

    final_df = df[[
        "model", "BERT_Precision", "BERT_Recall", "BERT_F1", "Final_Summary_Quality"
    ]].sort_values(by="Final_Summary_Quality", ascending=False)

    print("\n=== FINAL RESULTS ===")
    print(final_df.to_string(index=False))

    final_df.to_csv("scored_models.csv", index=False)
    print("\nSaved: scored_models.csv")

if __name__ == "__main__":
    main()