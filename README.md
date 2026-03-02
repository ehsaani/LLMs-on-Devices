<p align="center">
  <img src="https://github.com/eziyoo/LLMs-on-Devices/raw/main/figures/app_icon.png" alt="App Icon" width="200">
</p>

# 🌱 Sustainability Is Not Linear!

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)]()
[![Platform: Android](https://img.shields.io/badge/Platform-Android_16-green.svg)]()

> **Note:** This repository contains the experimental pipeline and configuration scripts for the paper: *"Sustainability Is Not Linear: Quantifying Performance, Energy, and Privacy Trade-offs in On-Device Intelligence"*. This work was authored by Eziyo Ehsani, Luca Giamattei, Ivano Malavolta, and Roberto Pietrantuono.

## 📖 Overview
Deploying Large Language Models (LLMs) on mobile devices promises enhanced privacy, low latency, and offline accessibility, but is fundamentally constrained by limited memory, thermal headroom, and battery capacity. 

This project provides a reproducible experimental pipeline to systematically evaluate the trade-offs between energy consumption, inference latency, memory footprint, and generation quality for on-device LLMs. It utilizes a non-intrusive energy profiling approach based on Android's BatteryManager API, testing eight open-source models ranging from 0.5B to 9B parameters.

## 🚀 Key Findings
* **The Quantization-Energy Paradox:** While importance-aware quantization (IQ4_XS) significantly reduces peak memory, it does not consistently reduce end-to-end energy on CPU-based inference compared to mixed-precision formats (Q4_K_M). De-quantization overhead can offset memory bandwidth savings.
* **Architecture > Parameter Count:** Model architecture and active computation per token are stronger predictors of on-device energy and latency than the specific 4-bit quantization variant. 
* **The Promise of Sparsity:** Mid-sized and sparse models (e.g., Mixture-of-Experts) achieve favorable quality-per-joule trade-offs compared to larger dense counterparts.
* **Metric Bias:** Reference-based evaluation metrics (BERTScore) exhibit extractive bias in this setting, occasionally favoring smaller models that copy input text. Reference-free LLM-as-a-judge protocols (G-Eval) better reflect abstractive quality and coherence.

## 🛠 Experimental Setup
### Hardware
* **Device:** Samsung Galaxy S25 Ultra
* **SoC:** Qualcomm Snapdragon 8 Elite
* **RAM:** 12 GB
* **OS:** Android 16

### Software Stack
* **Inference Engine:** `llama.cpp` (CPU-only inference).
* **Orchestration:** `Experiment Runner` framework via Python.
* **Telemetry:** On-device Android BatteryManager API monitoring via Wireless ADB.

## ⚙️ Methodology & Pipeline
The core of this repository is the `RunnerConfig.py` script, which automates a strict, isolated experimental loop to ensure reproducible energy measurements on an unrooted device. 

The pipeline strictly enforces:
1. **Device State Control:** Forces the screen on at minimum brightness and disables background activity to prevent OS heuristics from skewing CPU power usage.
2. **Measurement Synchronization:** Starts the BatteryManager service with a fixed 2-second spin-up before inference and terminates it immediately after text generation to minimize capturing post-inference idle tail power.
3. **Energy Integration:** Captures voltage and current at 100ms intervals (10Hz), subtracts baseline idle power, and calculates net energy consumed (Joules) using trapezoidal integration.
4. **Thermal Management:** Enforces a 200-second cool-down period between runs to reduce thermal carryover and mitigate throttling effects.

## 📊 Evaluated Models
Models evaluated under `Q4_K_M` and `IQ4_XS` quantization schemes:
* Qwen2-0.5B
* Qwen2.5-1.5B
* Phi-2 (2.78B)
* Qwen2.5-3B
* OLMoE-1B-7B (6.919B)
* Qwen2.5-7B
* Meta-Llama-3.1-8B
* Gemma-2-9B

## 💻 Getting Started
### Prerequisites
* Python 3.10+
* Android platform-tools (`adb`) configured globally
* `llama.cpp` built for Android (AArch64)
* A target Android device connected via Wireless ADB

### Execution
1. Update `DEVICE_ID`, `LOCAL_LLAMA_BUILD`, and `LOCAL_MODEL_PATH` in `RunnerConfig.py` to match your local environment and device IP.
2. Run the experiment through your Experiment Runner framework.
3. The script will automatically push required binaries/models, execute the warmup sequence, and begin the iterative testing matrix, saving outputs and parsed power metrics to the `/results` directory.

## 🎓 Authors & Contact
**Eziyo Ehsani**
* MSc in Data Science, University of Naples Federico II
* [LinkedIn](https://www.linkedin.com/in/eziyo/)

**Co-Authors:** Luca Giamattei, Prof. Ivano Malavolta, Prof. Roberto Pietrantuono

## 📝 Citation
If you use this pipeline or our findings in your research, please consider citing our paper:
```bibtex
@article{ehsani2026sustainability,
  title={Sustainability Is Not Linear: Quantifying Performance, Energy, and Privacy Trade-offs in On-Device Intelligence},
  author={Eziyo Ehsani, Ivano Malavolta, Roberto Pietrantuono},
  year={2026},
  institution={University of Naples Federico II & Vrije Universiteit Amsterdam}
}