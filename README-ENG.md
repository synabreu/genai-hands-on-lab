# Generative AI Hands-on Lab

A collection of Korean-language Jupyter Notebooks designed for hands-on learning of generative AI and GPU-based deep learning systems. The labs progressively cover essential production topics, including LLM deployment, inference serving, quantization, reproducible training, GPU profiling, performance optimization, and model evaluation.

## Learning Path

```text
LLM Deployment
  → Inference Serving Patterns
  → Production Serving with vLLM
  → Quantization and Memory Optimization
  → Reproducible Training
  → System-Level Profiling
  → Operator-Level Profiling
  → Batch Size and Precision Optimization
  → LLM Evaluation and Benchmarking
```

## Lab Summary

### [Lab 1 — Deploying and Serving LLMs in Production](Lab%201-Deploy-Serve-llama8b-vllm-KOR.ipynb)

Compares basic sequential inference using Hugging Face `transformers.generate()` with vLLM-based serving. Using the Llama 3 8B Instruct model, the lab measures per-request latency, token generation speed, and GPU memory usage. It demonstrates why batching and optimized inference engines are necessary in production environments.

**Key Topics:** Llama 3 8B, Hugging Face Transformers, vLLM, Sequential Inference, Throughput, Latency, VRAM

---

### [Lab 2 — Inference Serving Patterns: Dynamic Batching, Throughput, and the Triton Inference Model](Lab%202-Inference%20Serving%20Patterns-KOR.ipynb)

Analyzes why a simple server that processes requests one at a time fails to utilize the GPU efficiently. A lightweight Mini Triton server is implemented in Python to demonstrate how dynamic batching works. The lab compares QPS and average, p95, and p99 latency under different concurrency levels. It also provides a foundation for understanding Triton Inference Server’s `config.pbtxt` and continuous batching.

**Key Topics:** Dynamic Batching, Triton Inference Server, QPS, p95/p99 Latency, Request Queue, Concurrency

---

### [Lab 3 — Production Serving with vLLM: PagedAttention, Continuous Batching, and Prefix Caching](Lab%203-vLLM%20Production%20Serving-KOR.ipynb)

Explores LLM-specific serving optimization techniques using the TinyLlama-1.1B model and the vLLM engine. The lab measures the effects of PagedAttention-based KV cache block management, decode-step continuous batching, and prefix caching for shared system prompts. It also covers an OpenAI-compatible API server, Kubernetes deployment, Prometheus metrics, and Horizontal Pod Autoscaler configurations.

**Key Topics:** vLLM, PagedAttention, KV Cache, Continuous Batching, Prefix Caching, OpenAI-Compatible API, Kubernetes

---

### [Lab 4 — LLM Quantization and Optimization: INT8 and NF4](Lab%204-Quantize%20%26%20Optimize%20LLMs-KOR.ipynb)

Loads the TinyLlama-1.1B model in FP16, INT8, and NF4 formats and compares GPU memory usage, generation latency, and perplexity. The lab applies quantization directly with bitsandbytes and evaluates the memory savings and quality-performance trade-offs involved in running models on consumer GPUs.

**Key Topics:** Quantization, bitsandbytes, INT8, NF4, QLoRA, VRAM, Perplexity, GPTQ, AWQ

---

### [Lab 5 — Reproducible Training](Lab%205-Reproducible_Training-KOR.ipynb)

Analyzes why GPU training results may differ even when the same code and random seed are used. The lab controls PyTorch deterministic settings, the cuDNN autotuner, the cuBLAS workspace, and DataLoader worker random states. It also measures the throughput cost of deterministic execution and generates a training-run hash that combines the code, data, hyperparameters, and software environment.

**Key Topics:** Reproducibility, Deterministic Algorithms, Random Seed, cuDNN, cuBLAS, DataLoader, Training Run Hash

---

### [Lab 6 — Nsight Systems Profiling](Lab%206-Nsight%20Systems%20Profiling-PC-KOR.ipynb)

Uses NVIDIA Nsight Systems and NVTX to analyze CPU and GPU execution flows on a unified timeline. The lab instruments data loading, forward propagation, backward propagation, and optimizer steps. It identifies a bottleneck in which the GPU waits for the CPU DataLoader and compares throughput before and after optimization. It also introduces `.nsys-rep` reports and analysis using `nsys stats`.

**Key Topics:** Nsight Systems, NVTX, CUDA Timeline, Kernel Launch Gap, H2D/D2H Transfer, GPU Idle Time

---

### [Lab 7 — PyTorch Profiler and TensorBoard](Lab%207-PyTorch%20Profiler-KOR.ipynb)

Uses PyTorch Profiler to identify operators with high execution time and large memory allocations inside a model. The lab analyzes `key_averages()` output, records CPU and CUDA activity, and examines execution timelines using Chrome Trace and TensorBoard Profiler. It also compares the system-level analysis provided by Nsight Systems with the model-level analysis provided by PyTorch Profiler.

**Key Topics:** PyTorch Profiler, TensorBoard, Operator Profiling, Memory Profiling, Chrome Trace, Kineto

---

### [Lab 8 — Batch Size and Numerical Precision Sweep](Lab%208-BatchSize_PrecisionSweep-KOR.ipynb)

Combines different batch sizes with FP32, FP16, and BF16 numerical precision to measure throughput, latency, peak GPU memory usage, and output quality. The lab identifies the knee point at which throughput gains begin to diminish and uses cosine similarity against FP32 output as an accuracy acceptance threshold. It explains how to select the most efficient production configuration within a fixed GPU memory budget.

**Key Topics:** Batch Size Sweep, Precision Sweep, FP32, FP16, BF16, Tensor Core, Knee Point, Cosine Similarity

---

### [Lab 9 — LLM Evaluation and Benchmarking](Lab%209-Evaluation%20%26%20Benchmarking%20LLMs-KOR.ipynb)

Provides a step-by-step introduction to both traditional language-model metrics and modern LLM evaluation methods using GPT-2 and GPT-2 Medium. The lab examines the meaning and limitations of perplexity, BLEU, and ROUGE, and compares model outputs side by side. It also implements an LLM-as-a-Judge evaluator and demonstrates how to detect and mitigate evaluation biases such as position bias.

**Key Topics:** Perplexity, BLEU, ROUGE, Model Comparison, LLM-as-a-Judge, Position Bias, Benchmarking

## Environment

Although individual lab requirements vary, the following environment is assumed:

- Python 3.10 or later
- Jupyter Notebook or JupyterLab
- An NVIDIA GPU environment with PyTorch and CUDA support
- Hugging Face Transformers
- Additional lab-specific packages such as vLLM, bitsandbytes, and TensorBoard
- The NVIDIA Nsight Systems `nsys` command-line tool for Lab 6

Lab 1 uses a large model and therefore requires a relatively large amount of GPU memory. The remaining labs use TinyLlama, GPT-2, or small synthetic models so that the core concepts can be tested on comparatively smaller GPUs.

## Usage

```bash
git clone https://github.com/synabreu/genai-hands-on-lab.git
cd genai-hands-on-lab
jupyter lab
```

Review the objectives, required packages, and expected results in the first explanatory cell of each Notebook, and then execute the cells in order.

## Notes

The Notebooks in this repository are educational materials for learning generative AI and GPU systems. Execution results may vary depending on the GPU generation, CUDA and PyTorch versions, driver version, model cache state, and other processes running concurrently.

## License

This project is distributed under the **Apache License 2.0**. See the `LICENSE` file in this repository for more information.
