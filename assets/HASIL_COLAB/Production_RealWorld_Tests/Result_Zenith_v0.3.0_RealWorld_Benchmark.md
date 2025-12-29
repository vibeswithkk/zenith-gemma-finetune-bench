# Benchmark Result: Zenith v0.3.0 vs PyTorch Native

**Date:** 2025-12-30
**Version:** Zenith v0.3.0
**Device:** Tesla T4 (Google Colab)

## Executive Summary

This document records the performance metrics of `pyzenith` v0.3.0 compared to native PyTorch `2.5.1+cu124`. The benchmark focuses on real-world scenarios: fine-tuning a language model (TinyLlama-1.1B) and performing text generation.

**Key Highlight:** Zenith achieved a **+69.21%** increase in Inference Throughput (TPS).

---

## 1. System Configuration

*   **GPU:** Tesla T4 (15102 MiB VRAM)
*   **CPU:** Intel Xeon (2 vCPUs)
*   **RAM:** 12.7 GB
*   **OS:** Linux (Ubuntu 22.04)
*   **Python:** 3.10.12
*   **PyTorch:** 2.5.1+cu124
*   **Zenith:** 0.3.0 (Backend: `zenith`)

---

## 2. Test Results

### A. Training Benchmark (Fine-Tuning)
*Task: 50 steps of SFT (Supervised Fine-Tuning) on Alpaca dataset.*

| Metric | PyTorch (Baseline) | Zenith (Optimized) | Delta |
| :--- | :--- | :--- | :--- |
| **Total Duration** | 20.97s | 20.43s | **-0.54s** |
| **Speedup** | - | - | **+2.59%** |
| **Peak VRAM** | 2.59 GB | 2.59 GB | **0.00 GB** |

### B. Inference Benchmark (Text Generation)
*Task: Generate 100 new tokens. Input: "The future of AI is..."*

| Metric | PyTorch (Baseline) | Zenith (Optimized) | Delta |
| :--- | :--- | :--- | :--- |
| **Throughput (TPS)** | 15.37 tok/s | 26.02 tok/s | **+10.65 tok/s** |
| **Latency per Token** | 65.06 ms | 38.43 ms | **-26.63 ms** |
| **Speedup** | - | - | **+69.21%** |

### C. Numerical Stability
*Task: Compare output logits between PyTorch and Zenith.*

| Metric | Value | Result |
| :--- | :--- | :--- |
| **Mean Squared Error (MSE)** | 0.000000 | **PASSED** (Identical Output) |

---

## 3. Conclusion

The benchmark confirms that Zenith v0.3.0 successfully registers and activates its custom backend. The optimizations are particularly effective for **Inference**, demonstrating a significant **1.7x speedup** over the eager PyTorch baseline. Training performance shows a modest gain (+2.6%), while numerical stability remains perfect.
