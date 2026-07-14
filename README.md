# Generative AI Hands-on Lab

생성형 AI와 GPU 기반 딥러닝 시스템을 실습 중심으로 학습하기 위한 한국어 Jupyter Notebook 모음이다. LLM 배포와 추론 서빙부터 양자화, 재현 가능한 학습, GPU 프로파일링, 성능 최적화, 모델 평가까지 운영 환경에서 필요한 핵심 주제를 단계적으로 다룬다. 

Tip: If you're English, read [this file!](README-ENG.md)


## 학습 경로

```text
LLM 배포
  → 추론 서빙 패턴
  → vLLM 운영 서빙
  → 양자화와 메모리 최적화
  → 재현 가능한 학습
  → 시스템 수준 프로파일링
  → 연산자 수준 프로파일링
  → 배치 크기·정밀도 최적화
  → LLM 평가와 벤치마킹
```

## Lab 요약

### [Lab 1 — 프로덕션 환경에서 LLM 배포 및 서빙](Lab%201-Deploy-Serve-llama8b-vllm-KOR.ipynb)

Hugging Face `transformers.generate()`를 이용한 단순 순차 추론과 vLLM 기반 서빙을 비교한다. Llama 3 8B Instruct 모델을 대상으로 요청별 지연 시간, 토큰 처리 속도, GPU 메모리 사용량을 측정하며, 운영 환경에서 배칭과 최적화된 추론 엔진이 필요한 이유를 확인한다.

**핵심 주제:** Llama 3 8B, Hugging Face Transformers, vLLM, 순차 추론, 처리량, 지연 시간, VRAM

---

### [Lab 2 — 추론 서빙 패턴: 동적 배칭, 처리량, Triton 추론 모델](Lab%202-Inference%20Serving%20Patterns-KOR.ipynb)

요청을 하나씩 처리하는 단순 서버가 GPU를 충분히 활용하지 못하는 이유를 분석한다. Python으로 간단한 Mini Triton 서버를 구현해 동적 배칭의 작동 원리를 이해하고, 동시 요청 수에 따른 QPS와 평균·p95·p99 지연 시간을 비교한다. Triton Inference Server의 `config.pbtxt`와 연속 배칭을 이해하기 위한 기초 실습이다.

**핵심 주제:** Dynamic Batching, Triton Inference Server, QPS, p95/p99 지연 시간, 요청 큐, 동시성

---

### [Lab 3 — vLLM 운영 서빙: PagedAttention, 연속 배칭, 접두어 캐싱](Lab%203-vLLM%20Production%20Serving-KOR.ipynb)

TinyLlama-1.1B 모델과 vLLM 엔진을 사용해 LLM 전용 서빙 최적화 기술을 실험한다. PagedAttention의 KV 캐시 블록 관리, 디코딩 단계별 연속 배칭, 공통 시스템 프롬프트를 재사용하는 접두어 캐싱의 효과를 측정한다. OpenAI 호환 API 서버, Kubernetes 배포, Prometheus 지표와 HPA 구성도 함께 다룬다.

**핵심 주제:** vLLM, PagedAttention, KV Cache, Continuous Batching, Prefix Caching, OpenAI-compatible API, Kubernetes

---

### [Lab 4 — LLM 양자화 및 최적화: INT8과 NF4](Lab%204-Quantize%20%26%20Optimize%20LLMs-KOR.ipynb)

TinyLlama-1.1B 모델을 FP16, INT8, NF4 형식으로 로드해 GPU 메모리 사용량, 생성 지연 시간, 퍼플렉서티를 비교한다. bitsandbytes를 이용한 양자화를 직접 적용하고, 소비자용 GPU에서 모델을 실행하기 위한 메모리 절감 효과와 품질·성능 간 트레이드오프를 확인한다.

**핵심 주제:** Quantization, bitsandbytes, INT8, NF4, QLoRA, VRAM, Perplexity, GPTQ, AWQ

---

### [Lab 5 — 재현 가능한 학습](Lab%205-Reproducible_Training-KOR.ipynb)

동일한 코드와 난수 시드를 사용해도 GPU 학습 결과가 달라질 수 있는 원인을 분석한다. PyTorch 결정론 설정, cuDNN 자동 튜너, cuBLAS 작업 공간, DataLoader 작업자 난수 상태를 제어하고, 결정론적 실행이 처리량에 미치는 비용을 측정한다. 코드·데이터·하이퍼파라미터·환경을 결합한 학습 실행 해시도 생성한다.

**핵심 주제:** Reproducibility, Deterministic Algorithms, Random Seed, cuDNN, cuBLAS, DataLoader, Training Run Hash

---

### [Lab 6 — Nsight Systems Profiling](Lab%206-Nsight%20Systems%20Profiling-PC-KOR.ipynb)

NVIDIA Nsight Systems와 NVTX를 사용해 CPU와 GPU의 실행 흐름을 타임라인으로 분석한다. 데이터 로딩, 순전파, 역전파, 옵티마이저 단계를 계측하고, CPU 데이터로더 때문에 GPU가 대기하는 병목을 찾아 최적화 전후 처리량을 비교한다. `.nsys-rep` 보고서와 `nsys stats` 기반 분석 방법을 익힌다.

**핵심 주제:** Nsight Systems, NVTX, CUDA Timeline, Kernel Launch Gap, H2D/D2H Transfer, GPU Idle Time

---

### [Lab 7 — PyTorch Profiler와 TensorBoard](Lab%207-PyTorch%20Profiler-KOR.ipynb)

PyTorch Profiler를 이용해 모델 내부에서 실행 시간이 긴 연산자와 메모리 사용량이 큰 연산자를 찾는다. `key_averages()` 결과를 분석하고, CPU·CUDA 활동을 기록하며, Chrome Trace와 TensorBoard Profiler에서 실행 타임라인을 확인한다. Nsight Systems의 시스템 수준 분석과 PyTorch Profiler의 모델 수준 분석 차이도 비교한다.

**핵심 주제:** PyTorch Profiler, TensorBoard, Operator Profiling, Memory Profiling, Chrome Trace, Kineto

---

### [Lab 8 — 배치 크기와 수치 정밀도 스윕](Lab%208-BatchSize_PrecisionSweep-KOR.ipynb)

배치 크기와 FP32·FP16·BF16 정밀도를 조합해 처리량, 지연 시간, 최대 GPU 메모리 사용량과 출력 품질을 측정한다. 처리량 증가가 둔화되는 무릎 지점을 찾고, FP32 출력과의 코사인 유사도를 정확도 통과 기준으로 사용한다. GPU 메모리 예산 안에서 가장 효율적인 운영 구성을 선택하는 방법을 다룬다.

**핵심 주제:** Batch Size Sweep, Precision Sweep, FP32, FP16, BF16, Tensor Core, Knee Point, Cosine Similarity

---

### [Lab 9 — LLM 평가 및 벤치마킹](Lab%209-Evaluation%20%26%20Benchmarking%20LLMs-KOR.ipynb)

GPT-2와 GPT-2 Medium을 대상으로 전통적인 언어 모델 지표부터 현대적인 LLM 평가 방식까지 단계적으로 실습한다. 퍼플렉서티, BLEU, ROUGE의 의미와 한계를 확인하고, 모델 출력을 나란히 비교한다. LLM-as-a-Judge를 구현해 순서 편향과 같은 평가 편향을 탐지하고 완화하는 방법을 학습한다.

**핵심 주제:** Perplexity, BLEU, ROUGE, Model Comparison, LLM-as-a-Judge, Position Bias, Benchmarking

## 실행 환경

실습별 요구사항은 다르지만 다음 환경을 기본으로 가정한다.

- Python 3.10 이상
- Jupyter Notebook 또는 JupyterLab
- PyTorch와 CUDA를 지원하는 NVIDIA GPU 환경
- Hugging Face Transformers
- vLLM, bitsandbytes, TensorBoard 등 실습별 추가 패키지
- Lab 6 실행 시 NVIDIA Nsight Systems의 `nsys` 명령줄 도구

대형 모델을 사용하는 Lab 1은 GPU 메모리 요구량이 높다. 나머지 실습은 TinyLlama, GPT-2 또는 소규모 합성 모델을 활용해 비교적 작은 GPU에서도 핵심 개념을 검증할 수 있도록 구성되어 있다.

## 사용 방법

```bash
git clone https://github.com/synabreu/genai-hands-on-lab.git
cd genai-hands-on-lab
jupyter lab
```

각 Notebook의 첫 번째 설명 셀에서 실습 목표, 필요 패키지, 예상 결과를 확인한 뒤 순서대로 셀을 실행한다.

## 참고

이 저장소의 Notebook은 생성형 AI 및 GPU 시스템 학습을 위한 교육용 자료다. 실행 결과는 GPU 세대, CUDA·PyTorch 버전, 드라이버, 모델 캐시 상태와 동시 실행 프로세스에 따라 달라질 수 있다.

## License

이 프로젝트는 **Apache License 2.0**에 따라 배포된다. 자세한 내용은 저장소의 `LICENSE` 파일을 참고한다.
