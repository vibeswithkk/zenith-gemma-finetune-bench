# Zenith Inference Speed Test: Tokens Per Second (TPS)

**Test Date:** Sun Dec 28 2025
**Notebook:** [Zenith_Inference_Speed_Test.ipynb](https://colab.research.google.com/github/vibeswithkk/zenith-gemma-finetune-bench/blob/main/Zenith_Inference_Speed_Test.ipynb)

## Objective
Measure the inference speed (generation speed) improvement when using **Zenith Compiler** compared to native PyTorch.

**Metrics:**
*   **Tokens Per Second (TPS):** Higher is better.
*   **Latency:** Time to generate text.

## Hardware Setup
*   **GPU:** Tesla T4 (15GB VRAM)
*   **Driver:** 550.54.15
*   **CUDA:** 12.4

## Results

**INFERENCE SCOREBOARD (5 Runs)**

| Metric | PyTorch (Baseline) | Zenith (Optimized) | Improvement |
| :--- | :--- | :--- | :--- |
| **Avg TPS** | 30.79 | 32.26 | **+4.78%** |

**Detailed Run Logs:**

*   **PyTorch Best Run:** 32.71 TPS
*   **Zenith Best Run:** 33.68 TPS

### Analysis
Zenith successfully optimized the inference generation process, resulting in a **4.78% improvement in Tokens Per Second** (from ~30.8 to ~32.3 TPS) on a Tesla T4 GPU. This confirms that the graph compilation and optimization provided by Zenith translates to real-world speedups in text generation tasks.

---
*Note: The raw logs indicate successful installation and compilation of the Zenith backend.*
