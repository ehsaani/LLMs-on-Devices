import subprocess

context_text = (
    "The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee "
    "in 1989. He was working at CERN, the European Organization for Nuclear "
    "Research, near Geneva, Switzerland. Berners-Lee created the Web to meet "
    "the demand for automatic information-sharing between scientists in "
    "universities and institutes around the world."
)


final_prompt = f"Summarize the following text.\n{context_text}\nOutput:"



cmd = (
    f"cd /data/local/tmp && "
    f"LD_LIBRARY_PATH=. ./llama-cli "
    f"-m Phi-2-iq4_xs.gguf "
    f"-p '{final_prompt}' "
    f"-st "
    f"-n 128 "
    f"-c 512 -t 8 --temp 0 "
    )

subprocess.run(["adb", "-s", "192.168.43.162:5555", "shell", cmd])

"""

    "qwen2-0_5b-instruct-q4_k_m.gguf"
    "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "phi-2.Q4_K_M.gguf",
    "qwen2.5-3b-instruct-q4_k_m.gguf",
    "qwen2.5-7b-instruct-q4_k_m.gguf",
    "OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf",
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "gemma-2-9b-it-Q4_K_M.gguf"

The World Wide Web (WWW) was created by British scientist Tim Berners-Lee in 1989 while working at CERN,
the European Organization for Nuclear Research, near Geneva, Switzerland. He developed the Web to facilitate
the automatic sharing of information between scientists in different universities and research institutes worldwide.


"""
context_text = (
    "The World Wide Web (WWW) was invented by British scientist Tim Berners-Lee "
    "in 1989. He was working at CERN, the European Organization for Nuclear "
    "Research, near Geneva, Switzerland. Berners-Lee created the Web to meet "
    "the demand for automatic information-sharing between scientists in "
    "universities and institutes around the world."
)