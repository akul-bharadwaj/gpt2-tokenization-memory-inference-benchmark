# GPT-2 Tokenization, Memory Analysis & Inference Benchmarking

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QPWqYnoHeMtQd9k4hRqoQTArIoMKP1An?usp=sharing)

This project explores the practical systems-level behaviour of GPT-2 style language models through tokenization analysis, parameter memory estimation, activation memory profiling, and CPU/GPU inference benchmarking.

The work was completed as part of the **Large Language Models: A Hands-On Approach** course by **CCE, IISc Bengaluru**.

## Project Overview

The project is divided into three main parts:

1. **Tokenization Trade-offs**
   - Implemented character-level and whitespace tokenizers from scratch.
   - Compared them with GPT-2 BPE tokenization using `tiktoken`.
   - Analysed vocabulary size, compression ratio, and average token length.
   - Demonstrated the trade-off between vocabulary size and sequence length.

2. **GPT-2 Parameter and Memory Analysis**
   - Computed parameter counts for GPT-2 Small, Medium, Large, and XL configurations.
   - Estimated FP32 and FP16 parameter memory.
   - Analysed activation memory during inference.
   - Identified the attention score tensor as the dominant activation memory component due to its O(T²) scaling.
   - Verified parameter scaling behaviour using log-log plots.

3. **CPU vs GPU Inference Profiling**
   - Benchmarked GPT-2 Small, Medium, and Large on CPU.
   - Migrated models to a Google Colab T4 GPU and measured acceleration.
   - Compared FP32 and FP16 inference.
   - Tested batch-size scaling for GPT-2 Medium.
   - Measured latency, throughput, VRAM usage, and speedup.
   - Documented GPU memory behaviour and OOM/failure-mode handling.

## Dataset

The text corpus used for tokenization and prompt generation is:

**Alice's Adventures in Wonderland** from Project Gutenberg  
URL: `https://www.gutenberg.org/cache/epub/11/pg11.txt`

## Key Concepts Covered

- Character-level tokenization
- Whitespace tokenization with punctuation handling
- GPT-2 Byte Pair Encoding using `tiktoken`
- Compression ratio and vocabulary-size trade-offs
- GPT-2 parameter census
- FP32 vs FP16 memory estimation
- Activation memory analysis
- O(T²) attention memory scaling
- CPU inference benchmarking
- GPU inference benchmarking on T4
- Mixed precision inference
- Batch-size scaling
- Throughput vs latency trade-offs
- VRAM usage and OOM handling

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Tiktoken
- Pandas
- NumPy
- Plotly
- psutil
- Google Colab with T4 GPU

## Repository Structure

```text
.
├── Tokenization_GPT-2_Analysis_and_GPU_Inference.ipynb
└── README.md
```

## Main Results Summary

### Tokenization

The project compares three tokenizers:

| Tokenizer | Vocabulary Size | Compression Ratio |
|---|---:|---:|
| Character-level | Smallest | Lowest |
| Whitespace | Medium | Better than character-level |
| GPT-2 BPE | Largest | Most compact among tested methods |

<img width="1000" height="500" alt="newplot (5)" src="https://github.com/user-attachments/assets/967aff56-dfa5-4d50-a94b-103da9a90714" />

Character-level tokenization preserves all information but produces the longest sequences. GPT-2 BPE provides a better balance by representing frequent words and subwords compactly.

### Parameter Memory

GPT-2 model sizes were analysed using the official-style configurations:

| Model | Layers | d_model | d_ff | Approx. Size |
|---|---:|---:|---:|---:|
| Small | 12 | 768 | 3072 | 124M |
| Medium | 24 | 1024 | 4096 | 355M |
| Large | 36 | 1280 | 5120 | 774M |
| XL | 48 | 1600 | 6400 | 1.5B |

The calculations show how parameter memory scales with model size and why fixed embedding matrices affect scaling ratios between model families.

### Parameter Count vs FP16 Memory

The log-log plot below shows that FP16 memory scales approximately linearly with parameter count. This is expected because FP16 uses 2 bytes per parameter, making memory directly proportional to the number of parameters.

<img width="800" height="500" alt="newplot (4)" src="https://github.com/user-attachments/assets/afb350f7-e217-432f-a109-793f70465391" />

### Inference Benchmarking

The project benchmarks inference across CPU and T4 GPU environments. Metrics include:

- First-token latency
- Throughput in tokens/sec
- Peak RAM/VRAM usage
- CPU-to-GPU speedup
- FP32 vs FP16 performance
- Batch-size scaling behaviour

The experiments show that GPU speedup does not scale linearly with model size. Smaller models are more affected by fixed GPU overhead, while larger models better utilize GPU parallelism.

#### Final Inference Comparison

| Configuration | Device | Precision | Batch | Tok/sec | VRAM (GB) | Speedup vs CPU |
|---|---|---|---:|---:|---:|---:|
| Small 124M | CPU | FP32 | 1 | 9.70 | N/A | 1.00x |
| Small 124M | T4 | FP32 | 1 | 100.48 | 0.80 | 10.36x |
| Small 124M | T4 | FP16 | 1 | 110.54 | 1.72 | 11.40x |
| Medium 355M | T4 | FP16 | 8 | 388.78 | 2.73 | 79.72x |
| Large 774M | T4 | FP16 | 4 | 138.63 | 3.47 | 57.79x |

#### Medium 355M Batch Size Scaling

The Medium 355M model was benchmarked across batch sizes `[1, 2, 4, 8, 16]`. Throughput improved as batch size increased, while VRAM usage and per-step latency also increased.

<img width="850" height="500" alt="newplot (6)" src="https://github.com/user-attachments/assets/d25086a9-6313-4e1a-a200-f42ad77482fe" />

## How to Run

For GPU benchmarking, it is recommended to run the notebook in **Google Colab with a T4 GPU**.

## Notes

- CPU inference can be slow for GPT-2 Medium and Large.
- GPU timing can contain noise, especially on shared Colab environments.
- The benchmarking cells use warm-up iterations and averaged timed runs for more stable results.
- Measured throughput and VRAM values may vary across different Colab sessions.

## Learning Outcomes

This project helped me understand how LLM inference performance depends not only on model size, but also on tokenization, sequence length, memory layout, precision, batching, and hardware utilization.

It also helped connect theoretical model scaling concepts with practical deployment concerns such as latency, throughput, VRAM usage, and out-of-memory behaviour.

