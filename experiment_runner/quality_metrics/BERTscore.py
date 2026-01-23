import pandas as pd
import io
import numpy as np
from bert_score import score

# ==========================================
# 1. LOAD DATA
# ==========================================


# I have pasted your CSV data into this string variable for easy loading.
csv_data = """index,model,response
0,Qwen2-0.5B-Instruct,"Tim Berners-Lee, a British computer scientist, invented the World Wide Web in 1989. He drafted his initial vision in a document titled Information Management: A Proposal to solve a specific logistical problem. He created the first web page editor and server on a NeXT computer. He created the World Wide Web to meet the urgent demand for automatic information-sharing between scientists in universities and institutes around the world, allowing them to instantly link and access data regardless of their location. In 1"
1,Qwen2.5-1.5B-Instruct,"The World Wide Web (WWW) was conceptualized and invented by Tim Berners-Lee in 1989 while he was employed as a software engineer at CERN, the European Organization for Nuclear Research. Berners-Lee drafted his initial vision in a document titled Information Management: A Proposal to solve a specific logistical problem: the facility was so large that valuable data was constantly being lost or trapped in incompatible computers. By late 1990, he had specified the three fundamental technologies"
2,Phi-2-GGUF-iq4,"The World Wide Web (WWW) was created by Tim Berners-Lee, a British computer scientist, in 1989. Berners-Lee was working at CERN, a European Organization for Nuclear Research, when he came up with the idea. He wanted to solve the problem of lost or trapped data in the large facility. Berners-Lee developed HTML, URI, and HTTP, which are the three fundamental technologies of the Web. He also wrote the first web page editor and server on a"
3,Qwen2.5-3B-Instruct,"The World Wide Web (WWW) was conceptualized and invented by Tim Berners-Lee, a British computer scientist, in 1989 while he was employed at CERN. Initially, Berners-Lee aimed to solve a logistical problem at CERN, where valuable data was often lost or trapped in incompatible computers. He developed three fundamental technologies: HTML, URI, and HTTP, which remain the core of the Web today. Berners-Lee also created the first web page editor and"
4,Qwen2.5-7B-Instruct,"The World Wide Web (WWW) was conceptualized and invented by Tim Berners-Lee in 1989 while he was working as a software engineer at CERN. Berners-Lee aimed to solve the problem of data loss and incompatibility by creating a system for automatic information sharing. By late 1990, he had developed the fundamental technologies of HTML, URI, and HTTP. He also created the first web page editor and server on a NeXT computer. The"
5,OLMoE-1B-7B-Instruct,"The World Wide Web (WWW) was invented by British computer scientist Tim Berners-Lee in 1989 while he was working at CERN, a European Organization for Nuclear Research. Berners-Lee developed the initial vision for the WWW to address a logistical issue at CERN: the facility was so large that valuable data was frequently lost or inaccessible due to incompatible computers. He outlined three fundamental technologies that remain foundational to the WWW: HTML (Hypertext Markup Language), which defines"
6,Meta-Llama-3-8B-Instruct,"Here's a summary of the given text: British computer scientist Tim Berners-Lee conceptualized and invented the World Wide Web in 1989 while working at CERN. He aimed to solve a logistical problem of lost data by creating a system for automatic information-sharing between scientists worldwide. Berners-Lee developed the fundamental technologies of HTML, URI, and HTTP, and created the first web page editor and server. In 1993, CERN released the World Wide Web software into the public domain"
7,gemma-2-9b-it,"Tim Berners-Lee invented the World Wide Web (WWW) in 1989 while working at CERN.  He created it to solve the problem of information sharing between scientists at different institutions. Berners-Lee developed three key technologies: HTML, URI, and HTTP, which form the basis of the Web today. He also wrote the first web page editor and server. CERN released the Web software publicly in 1993, leading to the global information revolution"
"""
# Read the string into a pandas DataFrame
df = pd.read_csv(io.StringIO(csv_data))

# ==========================================
# 2. DEFINE GROUND TRUTH
# ==========================================
ground_truth_text = (
    "The World Wide Web (WWW) was conceptualized and invented by the British "
    "computer scientist Tim Berners-Lee in 1989. At the time, he was employed "
    "as a software engineer at CERN, the prestigious European Organization for "
    "Nuclear Research, located on the border of France and Geneva, Switzerland. "
    "Berners-Lee drafted his initial vision in a document titled Information "
    "Management: A Proposal to solve a specific logistical problem: the facility "
    "was so large that valuable data was constantly being lost or trapped in "
    "incompatible computers. By late 1990, he had specified the three fundamental "
    "technologies that remain the foundation of the Web today: HTML, URI, and "
    "HTTP. He also wrote the first web page editor and server on a NeXT computer. "
    "He created the Web to meet the urgent demand for automatic information-sharing "
    "between scientists in universities and institutes around the world, allowing "
    "them to instantly link and access data regardless of their location. In "
    "1993, CERN put the World Wide Web software in the public domain, sparking "
    "the global information revolution we know today."
)

# Prepare lists for BERTScore
candidates = df['response'].tolist()
references = [ground_truth_text] * len(candidates)

# ==========================================
# 3. CALCULATE METRICS
# ==========================================

print("1. Calculating BERTScore (Downloading model if needed)...")
# We grab Precision (P), Recall (R), and F1
P, R, F1 = score(candidates, references, lang="en", verbose=True)

# Add raw scores to DataFrame
df['BERT_Precision'] = P.numpy()
df['BERT_Recall'] = R.numpy()
df['BERT_F1'] = F1.numpy()

print("2. Calculating Length Penalty...")

# Calculate Word Counts
ref_word_count = len(ground_truth_text.split())
df['response_word_count'] = df['response'].apply(lambda x: len(str(x).split()))

# Calculate Compression Ratio (Response Length / Source Length)
# 1.0 = Same length (Copy Paste), 0.5 = Half length (Summary)
df['compression_ratio'] = df['response_word_count'] / ref_word_count

# ==========================================
# 4. APPLY "PARROT PENALTY"
# ==========================================
# Logic: 
# If a model copied more than 85% of the text (ratio > 0.85), it failed the summary task.
# If it summarized successfully (ratio <= 0.85), we use its Precision score (how accurate was the summary?)

def calculate_final_score(row):
    ratio = row['compression_ratio']
    precision = row['BERT_Precision']
    
    # PENALTY THRESHOLD
    if ratio > 0.85: 
        return precision  # Zero score for copying
    else:
        return precision # Return the accuracy of the summary

df['Final_Summary_Quality'] = df.apply(calculate_final_score, axis=1)

# ==========================================
# 5. DISPLAY RESULTS
# ==========================================

# Sort by the new Final Score
final_df = df[[
    'model', 
    'compression_ratio', 
    'BERT_Precision', 
    'Final_Summary_Quality'
]].sort_values(by='Final_Summary_Quality', ascending=False)

print("\n=== FINAL RESULTS (Ranked by Summary Quality) ===")
print(final_df)

# Optional: Save to CSV
final_df.to_csv("scored_models.csv", index=False)