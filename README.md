# Multimodal Vision-Language Model

A from-scratch implementation of a Vision-Language Model (VLM) that fuses a SigLIP-based vision encoder with a Gemma-style causal language model through cross-attention and Perceiver Resampling. Trained end-to-end for visual instruction following with support for contrastive alignment, Mixture-of-Experts routing, speculative decoding, paged KV-cache inference, RAG-augmented chat, and distributed FSDP training.

Built entirely in PyTorch — no pretrained VLM weights, no third-party model hubs. Everything from patch embeddings to top-p sampling is written from the ground up.

---

## Table of Contents

- [Architecture](#architecture)
- [System Design](#system-design)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Docker](#docker)
- [Training](#training)
- [Inference](#inference)
- [Evaluation & Benchmarks](#evaluation--benchmarks)
- [Retrieval-Augmented Generation](#retrieval-augmented-generation)
- [Agent System](#agent-system)
- [Distributed Training](#distributed-training)
- [Configuration](#configuration)
- [Pipeline](#pipeline)
- [Results](#results)
- [Key Design Decisions](#key-design-decisions)
- [Acknowledgments](#acknowledgments)

---

## Architecture

### End-to-End Model Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#e8f4fd', 'primaryBorderColor': '#3498db', 'lineColor': '#2c3e50', 'fontSize': '14px'}}}%%
graph TD
    subgraph INPUT ["📥 Input"]
        direction LR
        IMG["Image<br/>3 × 224 × 224"]
        TXT["Text Tokens<br/>[B, T]"]
    end

    subgraph VISION ["👁️ Vision Encoder"]
        direction TB
        PE["<b>Patch Embedding</b><br/>Conv2d(3→768, k=16, s=16)<br/>196 patches + CLS token<br/>+ positional embeddings"]
        SE["<b>SigLIP Encoder</b><br/>12 × TransformerBlock<br/>768-d · 12 heads<br/>gradient checkpointing"]
        PR["<b>Perceiver Resampler</b><br/>64 learnable latent queries<br/>cross-attention (8 heads)<br/>→ [B, 64, 768]"]
        TC["<b>Token Compressor</b><br/>32 learnable queries<br/>MultiheadAttention (8 heads)<br/>→ [B, 32, 768]"]
        IP["<b>Image Projector</b><br/>Linear(768 → 768)"]
        PE --> SE --> PR --> TC --> IP
    end

    subgraph LLM ["🧠 Language Model — Gemma (12 layers)"]
        direction TB
        EMB["<b>Token Embedding</b><br/>Embed(50257, 768)"]
        DL["<b>12 × Decoder Layer</b><br/>┌─ GQA Self-Attention (8 heads)<br/>├─ Cross-Attention (Q=text, KV=vision)<br/>└─ MoE FFN (4 experts, top-2)"]
        RN["<b>RMSNorm</b>"]
        LH["<b>LM Head</b><br/>Linear(768 → 50257)<br/>weight-tied with embedding"]
        EMB --> DL --> RN --> LH
    end

    subgraph HEADS ["📐 Contrastive Projection"]
        direction LR
        IH["Image Head<br/>768 → GELU → 512"]
        TH["Text Head<br/>768 → GELU → 512"]
    end

    IMG --> PE
    TXT --> EMB
    IP -- "32 vision<br/>tokens" --> DL
    RN -. "hidden states" .-> IH
    RN -. "hidden states" .-> TH
    LH --> OUTPUT["logits<br/>[B, T, 50257]"]
    IH --> CLOSS["Contrastive<br/>Loss"]
    TH --> CLOSS

    style IMG fill:#4a9eff,stroke:#2980b9,color:#fff,stroke-width:2px
    style TXT fill:#4a9eff,stroke:#2980b9,color:#fff,stroke-width:2px
    style OUTPUT fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
    style CLOSS fill:#e74c3c,stroke:#c0392b,color:#fff,stroke-width:2px
    style PE fill:#ebf5fb,stroke:#3498db,stroke-width:1px
    style SE fill:#ebf5fb,stroke:#3498db,stroke-width:1px
    style PR fill:#ebf5fb,stroke:#3498db,stroke-width:1px
    style TC fill:#ebf5fb,stroke:#3498db,stroke-width:1px
    style IP fill:#ebf5fb,stroke:#3498db,stroke-width:1px
    style EMB fill:#fdf2e9,stroke:#e67e22,stroke-width:1px
    style DL fill:#fdf2e9,stroke:#e67e22,stroke-width:1px
    style RN fill:#fdf2e9,stroke:#e67e22,stroke-width:1px
    style LH fill:#fdf2e9,stroke:#e67e22,stroke-width:1px
```

### Decoder Layer — Internal Structure

Each of the 12 decoder layers applies three sequential operations with residual connections and pre-normalization:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#fff', 'lineColor': '#2c3e50'}}}%%
graph TD
    INPUT["Input Hidden States<br/>[B, T, 768]"] --> SPLIT1{" "}

    SPLIT1 -->|"residual"| ADD1
    SPLIT1 --> NORM1["RMSNorm"]
    NORM1 --> GQA["<b>GQA Self-Attention</b><br/>8 query heads · dim 96/head<br/>Grouped Key-Value sharing<br/>Causal mask (training)<br/>KV-Cache (inference)"]
    GQA --> ADD1(("⊕"))

    ADD1 --> SPLIT2{" "}
    SPLIT2 -->|"residual"| ADD2
    SPLIT2 --> NORM2["RMSNorm"]
    NORM2 --> CROSS["<b>Cross-Attention</b><br/>Q = text hidden states<br/>K, V = 32 vision tokens<br/>8 heads · dim 96/head"]
    CROSS --> ADD2(("⊕"))

    ADD2 --> SPLIT3{" "}
    SPLIT3 -->|"residual"| ADD3
    SPLIT3 --> NORM3["RMSNorm"]
    NORM3 --> ROUTER["<b>Router</b><br/>Linear(768→4)<br/>Softmax → Top-2"]
    ROUTER --> E1["Expert 1<br/>768→3072→768"]
    ROUTER --> E2["Expert 2<br/>768→3072→768"]
    ROUTER --> E3["Expert 3<br/>768→3072→768"]
    ROUTER --> E4["Expert 4<br/>768→3072→768"]
    E1 & E2 & E3 & E4 --> COMBINE["Weighted<br/>Combination"]
    COMBINE --> ADD3(("⊕"))

    ADD3 --> OUT["Output → Next Layer"]

    COMBINE -. "load balance<br/>loss" .-> AUX["MoE<br/>Aux Loss"]

    style GQA fill:#3498db,stroke:#2471a3,color:#fff,stroke-width:2px
    style CROSS fill:#9b59b6,stroke:#7d3c98,color:#fff,stroke-width:2px
    style ROUTER fill:#e67e22,stroke:#ca6f1e,color:#fff,stroke-width:2px
    style E1 fill:#f5cba7,stroke:#e67e22,stroke-width:1px
    style E2 fill:#f5cba7,stroke:#e67e22,stroke-width:1px
    style E3 fill:#f5cba7,stroke:#e67e22,stroke-width:1px
    style E4 fill:#f5cba7,stroke:#e67e22,stroke-width:1px
    style ADD1 fill:#27ae60,stroke:#1e8449,color:#fff
    style ADD2 fill:#27ae60,stroke:#1e8449,color:#fff
    style ADD3 fill:#27ae60,stroke:#1e8449,color:#fff
    style AUX fill:#e74c3c,stroke:#c0392b,color:#fff
    style INPUT fill:#d5f5e3,stroke:#27ae60,stroke-width:2px
    style OUT fill:#d5f5e3,stroke:#27ae60,stroke-width:2px
```

### Vision Token Compression Pipeline

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#2c3e50'}}}%%
graph LR
    A["Raw Image<br/>[3, 224, 224]"] ==> B["Patch Embed<br/>Conv2d 16×16"]
    B ==>|"[B, 226, 768]<br/>196 patches<br/>+ CLS + pos"| C["SigLIP<br/>12 blocks"]
    C ==>|"[B, 226, 768]"| D["Perceiver<br/>Resampler"]
    D ==>|"[B, 64, 768]<br/>64 latents"| E["Token<br/>Compressor"]
    E ==>|"[B, 32, 768]<br/>32 tokens"| F["Projector<br/>768→768"]
    F ==>|"[B, 32, 768]"| G["Cross-Attn<br/>in Decoder"]

    style A fill:#4a9eff,stroke:#2980b9,color:#fff,stroke-width:2px
    style B fill:#85c1e9,stroke:#2980b9,stroke-width:1px
    style C fill:#85c1e9,stroke:#2980b9,stroke-width:1px
    style D fill:#bb8fce,stroke:#7d3c98,stroke-width:1px
    style E fill:#bb8fce,stroke:#7d3c98,stroke-width:1px
    style F fill:#f0b27a,stroke:#e67e22,stroke-width:1px
    style G fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
```

> **Compression ratio:** 196 raw patches → 64 resampled → 32 compressed tokens (6× reduction)

---

## System Design

### End-to-End Training Workflow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#2c3e50', 'fontSize': '13px'}}}%%
flowchart TB
    subgraph CONFIG ["⚙️ Configuration"]
        direction LR
        M_CFG["model.yaml<br/>vision_dim: 768<br/>text_dim: 768<br/>num_layers: 12<br/>vocab_size: 50257"]
        T_CFG["training.yaml<br/>batch_size: 4<br/>epochs: 50<br/>contrastive_weight<br/>moe_weight"]
        O_CFG["optimizer.yaml<br/>lr: 5e-4<br/>weight_decay: 0.01<br/>betas: [0.9, 0.95]"]
    end

    subgraph DATA ["📂 Data Pipeline"]
        direction TB
        RAW["instruction_data.json<br/>+ data/images/"]
        DS["InstructionDataset<br/>GPT-2 BPE tokenizer<br/>Image: resize 224×224<br/>Labels: prompt=-100, answer=tokens"]
        DL["DataLoader<br/>batch_size=4, shuffle=True<br/>collate_fn (pad + stack)"]
        RAW --> DS --> DL
    end

    subgraph TRAIN ["🏋️ Training Loop"]
        direction TB
        FWD["Forward Pass<br/>logits, img_emb, txt_emb, moe_loss<br/>= VLM(images, tokens)"]
        LOSS["Loss Computation"]
        LM_L["LM Loss<br/>CE(logits[:,:-1], labels[:,1:])<br/>ignore_index=-100"]
        CON_L["Contrastive Loss<br/>normalize → sim matrix<br/>→ CE(sim, arange(B))"]
        MOE_L["MoE Aux Loss<br/>load balancing<br/>across 4 experts"]
        TOTAL["Total = LM + α·Contra + β·MoE"]
        BACK["Backward Pass<br/>clip_grad_norm_(1.0)<br/>AdamW step"]
        FWD --> LOSS
        LOSS --> LM_L & CON_L & MOE_L
        LM_L & CON_L & MOE_L --> TOTAL --> BACK
    end

    subgraph OUT ["💾 Output"]
        direction LR
        CKPT["Checkpoint<br/>model_state_dict<br/>optimizer_state_dict<br/>step"]
        LOGS["Logger<br/>train_log.json<br/>step + loss"]
    end

    CONFIG --> DS
    DL --> FWD
    BACK --> CKPT & LOGS
    BACK -->|"next batch"| FWD

    style LM_L fill:#e74c3c,stroke:#c0392b,color:#fff
    style CON_L fill:#3498db,stroke:#2471a3,color:#fff
    style MOE_L fill:#e67e22,stroke:#ca6f1e,color:#fff
    style TOTAL fill:#8e44ad,stroke:#6c3483,color:#fff
    style CKPT fill:#27ae60,stroke:#1e8449,color:#fff
    style LOGS fill:#27ae60,stroke:#1e8449,color:#fff
```

### Inference & Generation Workflow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#2c3e50', 'fontSize': '13px'}}}%%
flowchart LR
    subgraph PREP ["📥 Preprocessing"]
        direction TB
        IMG_IN["Image Path"] --> RESIZE["Resize 224×224<br/>ToTensor<br/>Normalize"]
        TXT_IN["User Prompt"] --> BUILD["build_prompt()<br/>'User: {q}<br/>Assistant:'"]
        BUILD --> TOKENIZE["GPT-2 BPE<br/>Tokenize"]
    end

    subgraph MODEL ["🧠 Model"]
        LOAD["Load VLM<br/>+ Checkpoint"]
    end

    subgraph DECODE ["🔄 Autoregressive Decoding"]
        direction TB
        STEP["VLM Forward<br/>(image, tokens, cache)"]
        LOGITS["Extract logits<br/>at last position"]
        TOPP["Top-p Sampling<br/>temp=0.8, p=0.9<br/>nucleus selection"]
        CHECK{"Token\n== EOS?"}
        APPEND["Append token<br/>to sequence"]
        CACHE["Paged KV-Cache<br/>12 layers × 8 heads<br/>16 tokens/page<br/>1024 max pages"]

        STEP --> LOGITS --> TOPP --> CHECK
        CHECK -->|"No"| APPEND --> STEP
        STEP <-.->|"read/write"| CACHE
    end

    subgraph OUTPUT ["📤 Output"]
        DETOK["Detokenize<br/>GPT-2 decode<br/>→ answer string"]
    end

    RESIZE --> LOAD
    TOKENIZE --> LOAD
    LOAD --> STEP
    CHECK -->|"Yes"| DETOK

    style IMG_IN fill:#4a9eff,stroke:#2980b9,color:#fff
    style TXT_IN fill:#4a9eff,stroke:#2980b9,color:#fff
    style DETOK fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
    style CACHE fill:#f39c12,stroke:#ca6f1e,color:#fff
    style TOPP fill:#9b59b6,stroke:#7d3c98,color:#fff
    style CHECK fill:#e67e22,stroke:#ca6f1e,color:#fff
```

### Chat System Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#2c3e50', 'fontSize': '13px'}}}%%
flowchart TB
    USER["👤 User<br/>Image + Question"] --> VLMCHAT["<b>VLMChat</b><br/>Orchestrator"]

    VLMCHAT --> RAG_BRANCH
    VLMCHAT --> AGENT_BRANCH
    VLMCHAT --> MEMORY

    subgraph RAG_BRANCH ["📚 Retrieval-Augmented Generation"]
        direction LR
        KB["Knowledge Base<br/>wiki.txt"] --> LOADER["load_knowledge()"]
        LOADER --> EMBEDDER["SimpleEmbedder<br/>GPT-2 tokenize<br/>Embedding lookup<br/>Mean pool → [768]"]
        EMBEDDER --> INDEX["Doc Vectors<br/>[N, 768]"]
        QUERY_EMB["Query → [1, 768]"] --> COSINE["Cosine<br/>Similarity"]
        INDEX --> COSINE
        COSINE --> TOPK["Top-k<br/>Passages"]
    end

    subgraph AGENT_BRANCH ["🤖 ReAct Agent"]
        direction TB
        THINK["Thought<br/>Analyze the task"]
        ACT["Action<br/>Select tool"]
        OBS["Observation<br/>Collect result"]
        THINK --> ACT --> OBS --> THINK
        ACT --> TOOLS
    end

    subgraph TOOLS ["🔧 Tool Router"]
        direction LR
        CALC["Calculator<br/>math expressions"]
        CAPTION["Captioner<br/>image description"]
    end

    subgraph MEMORY ["💭 Memory"]
        direction TB
        HIST["ConversationMemory<br/>Sliding window<br/>Max 5 turns<br/>User + Assistant pairs"]
    end

    TOPK -->|"retrieved<br/>context"| GEN
    AGENT_BRANCH -->|"final<br/>answer"| GEN
    MEMORY -->|"history<br/>context"| GEN

    GEN["<b>Generator</b><br/>VLM Inference"] --> ANSWER["💬 Model Response"]

    style USER fill:#4a9eff,stroke:#2980b9,color:#fff,stroke-width:2px
    style ANSWER fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
    style GEN fill:#8e44ad,stroke:#6c3483,color:#fff,stroke-width:2px
    style VLMCHAT fill:#2c3e50,stroke:#1a252f,color:#fff,stroke-width:2px
    style CALC fill:#f5cba7,stroke:#e67e22
    style CAPTION fill:#f5cba7,stroke:#e67e22
    style THINK fill:#aed6f1,stroke:#3498db
    style ACT fill:#f5cba7,stroke:#e67e22
    style OBS fill:#abebc6,stroke:#27ae60
```

### Evaluation Pipeline

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#2c3e50'}}}%%
flowchart LR
    subgraph LOAD ["Load"]
        CKPT["Trained<br/>Checkpoint"] --> MODEL["VLM<br/>model.eval()"]
        DATA["Test Data"] --> DLOADER["DataLoader<br/>batch_size=2"]
    end

    subgraph EVAL ["Evaluation Tasks"]
        direction TB
        CAP["<b>Caption Evaluation</b><br/>Generate captions<br/>Compare vs reference<br/>→ BLEU score (NLTK)"]
        VQA["<b>VQA Evaluation</b><br/>Predict answer token<br/>Compare vs ground truth<br/>→ Exact-match accuracy"]
        RET["<b>Retrieval Evaluation</b><br/>Image & text embeddings<br/>Cosine similarity matrix<br/>→ Recall@K"]
    end

    MODEL --> CAP & VQA & RET
    DLOADER --> CAP & VQA & RET

    CAP --> REPORT
    VQA --> REPORT
    RET --> REPORT

    REPORT["📊 Benchmark Report<br/>BLEU | Accuracy | Recall@K"]

    style CKPT fill:#f39c12,stroke:#ca6f1e,color:#fff
    style REPORT fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
    style CAP fill:#ebf5fb,stroke:#3498db,stroke-width:2px
    style VQA fill:#fdf2e9,stroke:#e67e22,stroke-width:2px
    style RET fill:#f4ecf7,stroke:#9b59b6,stroke-width:2px
```

---

## Project Structure

```
├── configs/
│   ├── model.yaml               # Model architecture (dims, layers, heads, vocab)
│   ├── training.yaml            # Training hyperparams (batch, epochs, loss weights)
│   └── optimizer.yaml           # AdamW settings (lr, betas, weight decay)
│
├── vision/
│   ├── patch_embedding.py       # Conv2d patch tokenizer + positional encoding
│   ├── attention.py             # Multi-head self-attention for vision
│   ├── transformer_block.py     # Vision transformer block w/ gradient checkpointing
│   ├── siglip_encoder.py        # 12-layer SigLIP vision encoder
│   ├── perceiver_resampler.py   # Cross-attention resampler (→ 64 latent tokens)
│   └── token_compressor.py      # MHA-based compressor (→ 32 tokens)
│
├── text/
│   ├── gemma_model.py           # 12-layer Gemma LM with weight-tied LM head
│   ├── decoder_layer.py         # Self-attn → Cross-attn → MoE FFN
│   ├── gqa_attention.py         # Grouped Query Attention with KV-cache support
│   ├── cross_attention.py       # Text-to-vision cross-attention
│   ├── ffn.py                   # Standard feedforward (768 → 3072 → 768)
│   ├── moe_ffn.py               # Mixture-of-Experts (4 experts, top-2 routing)
│   ├── rmsnorm.py               # RMS layer normalization
│   ├── rotary.py                # Rotary positional embeddings (RoPE)
│   └── lora.py                  # LoRA adapters (rank 8, alpha 16)
│
├── multimodal/
│   ├── vlm_model.py             # End-to-end VLM orchestrator
│   ├── projector.py             # Vision → text space projection
│   └── projection_heads.py      # Contrastive learning projection heads
│
├── training/
│   ├── train_vlm.py             # Main training loop (LM + contrastive + MoE loss)
│   ├── trainer.py               # Lightweight trainer wrapper class
│   ├── train_siglip.py          # Standalone SigLIP contrastive training
│   └── contrastive_loss.py      # BCE-based contrastive loss
│
├── dataset/
│   ├── instruction_dataset.py   # Instruction-following dataset with GPT-2 tokenizer
│   ├── instruction_format.py    # Prompt template builder
│   ├── preprocessing.py         # Token padding utilities
│   ├── tokenizer.py             # tiktoken wrapper
│   └── webdataset_loader.py     # WebDataset streaming loader for large-scale data
│
├── inference/
│   ├── generate.py              # Autoregressive generation with top-p sampling
│   ├── chat_vlm.py              # RAG-augmented chat interface
│   ├── run_chat.py              # Interactive chat entry point
│   ├── sampling.py              # Nucleus (top-p) sampling
│   ├── kv_cache.py              # Basic KV-cache
│   ├── paged_kv_cache.py        # Paged KV-cache for long sequences
│   └── speculative_decoder.py   # Speculative decoding (draft + verify)
│
├── evaluation/
│   ├── evaluate.py              # Caption, VQA, and retrieval evaluators
│   ├── run_benchmarks.py        # Automated benchmark runner
│   ├── caption_metrics.py       # BLEU score (via NLTK)
│   ├── vqa_metrics.py           # Exact-match VQA accuracy
│   ├── retrieval_metrics.py     # Recall@K
│   └── report.py                # Metric aggregation
│
├── agents/
│   ├── react_agent.py           # ReAct reasoning agent (Thought → Action → Observe)
│   ├── router.py                # Tool routing by keyword detection
│   ├── tools.py                 # Calculator + captioner tools
│   └── memory.py                # Sliding-window conversation memory (5 turns)
│
├── retrieval/
│   ├── retriever.py             # Top-k cosine similarity retrieval
│   ├── embedder.py              # GPT-2 token embedding + mean pooling
│   └── load_knowledge.py        # Plain-text knowledge base loader
│
├── distributed/
│   ├── fsdp.py                  # Fully Sharded Data Parallel wrapper
│   └── launch.py                # Multi-GPU process launcher (mp.spawn)
│
├── experiments/
│   └── logger.py                # JSON training logger
│
├── utils/
│   ├── config.py                # YAML config loader (safe_load)
│   └── seed.py                  # Reproducibility seeding
│
├── scripts/
│   ├── run_all.sh               # Full 9-step pipeline
│   ├── run_train.sh             # Training launcher
│   ├── run_inference.sh         # Inference launcher
│   ├── run_chat.sh              # Chat launcher
│   ├── run_experiment.sh        # Experiment with timestamped output
│   └── run_pipeline.sh          # Interactive step selector
│
├── data/
│   ├── instruction_data.json    # Instruction-following training data
│   └── images/                  # Training images
│
├── knowledge/
│   └── wiki.txt                 # Knowledge base for RAG retrieval
│
├── Dockerfile                   # GPU-enabled container image
├── docker-compose.yml           # Multi-service orchestration
├── .dockerignore                # Build context exclusions
├── demo.py                      # Quick demo script
└── requirements.txt             # Python dependencies
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested on NVIDIA RTX A6000)
- ~8 GB VRAM minimum

### Installation

```bash
git clone https://github.com/<your-username>/Multimodal_Vision-Language-Model_Research.git
cd Multimodal_Vision-Language-Model_Research

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Verify GPU

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Docker

The project ships with a `Dockerfile` and `docker-compose.yml` for containerized training and inference. Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU passthrough.

### Build

```bash
docker compose build
```

### Docker Services

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#2c3e50', 'fontSize': '13px'}}}%%
flowchart TB
    subgraph HOST ["🖥️ Host Machine"]
        direction LR
        DATA_VOL["📁 data/<br/>images + json"]
        OUT_VOL["💾 outputs/<br/>checkpoints"]
        CFG_VOL["⚙️ configs/<br/>yaml files"]
        KB_VOL["📚 knowledge/<br/>wiki.txt"]
    end

    GPU["NVIDIA GPU<br/>Container Toolkit<br/>CUDA 12.8"] --> COMPOSE

    subgraph COMPOSE ["🐳 Docker Compose Services"]
        direction TB
        TRAIN["<b>train</b><br/>python -m training.train_vlm<br/>Full model training"]
        INFER["<b>inference</b><br/>python -m inference.generate<br/>Interactive (stdin/tty)"]
        CHAT["<b>chat</b><br/>python -m inference.run_chat<br/>RAG + Agent tools"]
        EVAL["<b>evaluate</b><br/>python -m evaluation.run_benchmarks<br/>BLEU · VQA · Recall@K"]
        DEMO["<b>demo</b><br/>python -m demo<br/>Single-pass test"]
    end

    DATA_VOL -.-> TRAIN & INFER & CHAT & EVAL & DEMO
    CFG_VOL -.-> TRAIN
    KB_VOL -.-> CHAT
    TRAIN -->|"saves"| OUT_VOL
    INFER & CHAT & EVAL & DEMO -->|"loads"| OUT_VOL

    style GPU fill:#76b900,stroke:#4a7a00,color:#fff,stroke-width:2px
    style TRAIN fill:#e74c3c,stroke:#c0392b,color:#fff
    style INFER fill:#3498db,stroke:#2471a3,color:#fff
    style CHAT fill:#9b59b6,stroke:#7d3c98,color:#fff
    style EVAL fill:#e67e22,stroke:#ca6f1e,color:#fff
    style DEMO fill:#1abc9c,stroke:#16a085,color:#fff
    style OUT_VOL fill:#f39c12,stroke:#ca6f1e,color:#fff,stroke-width:2px
```

### Usage

```bash
# Train the model
docker compose run train

# Interactive inference (prompts for image path + question)
docker compose run inference

# Multi-turn chat with RAG and agent tools
docker compose run chat

# Run evaluation benchmarks
docker compose run evaluate

# Quick demo
docker compose run demo
```

All services mount `data/` and `outputs/` as volumes, so checkpoints trained inside the container persist on the host. The `chat` service additionally mounts `knowledge/` for the RAG retrieval pipeline.

### Dockerfile Details

| Property | Value |
|----------|-------|
| **Base image** | `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04` |
| **Python** | 3.12 |
| **NLTK data** | Pre-downloaded inside the image |
| **CUDA config** | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

### Custom Training with Docker

```bash
# Train with custom config
docker compose run train python -m training.train_vlm \
    --model_config configs/model.yaml \
    --train_config configs/training.yaml \
    --output_dir outputs

# Train SigLIP only
docker compose run train python -m training.train_siglip
```

---

## Training

### Quick Start

```bash
python -m training.train_vlm
```

This loads configs from `configs/` and trains the full VLM with the default settings.

### Custom Training

```bash
python -m training.train_vlm \
    --model_config configs/model.yaml \
    --train_config configs/training.yaml \
    --optim_config configs/optimizer.yaml \
    --output_dir outputs
```

### Training Details

The training loop optimizes a weighted combination of three objectives:

| Loss | Purpose | Weight |
|------|---------|--------|
| **Language Modeling** | Next-token prediction with shifted labels. Prompt tokens are masked (`-100`) so only the answer is supervised. | 1.0 |
| **Contrastive** | Symmetric cross-entropy over image–text cosine similarity matrix. Aligns vision and text embeddings. | configurable |
| **MoE Auxiliary** | Load-balancing loss that encourages uniform expert utilization across the 4 MoE experts. | configurable |

**Optimizer:** AdamW with `lr=5e-4`, `weight_decay=0.01`, `betas=(0.9, 0.95)`, and gradient clipping at `max_norm=1.0`.

**CUDA Optimizations:** TF32 matmul is enabled (`allow_tf32 = True`) and memory allocation uses expandable segments to reduce fragmentation.

Checkpoints are saved periodically and at the end of training. Training automatically resumes from the latest checkpoint if one exists in the output directory.

### Multi-Objective Loss Computation

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#2c3e50'}}}%%
flowchart TD
    FWD["<b>VLM Forward</b><br/>logits, img_emb, txt_emb, moe_loss"] --> BRANCH1 & BRANCH2 & BRANCH3

    subgraph BRANCH1 ["Language Modeling"]
        SHIFT["Shift Alignment<br/>logits[:, :-1]<br/>labels[:, 1:]"] --> CE["CrossEntropy<br/>ignore_index=-100"]
    end

    subgraph BRANCH2 ["Contrastive Alignment"]
        NORMI["L2 Normalize<br/>img_emb"] --> SIM["Similarity<br/>img @ txt.T"]
        NORMT["L2 Normalize<br/>txt_emb"] --> SIM
        SIM --> CCE["CrossEntropy<br/>targets=arange(B)"]
    end

    subgraph BRANCH3 ["MoE Regularization"]
        AUX["Load Balance Loss<br/>N × Σ(prob_mean × token_frac)"]
    end

    CE --> TOTAL
    CCE --> TOTAL
    AUX --> TOTAL

    TOTAL["<b>Total Loss</b><br/>= LM + α·Contrastive + β·MoE"] --> BACK["Backward → Clip(1.0) → AdamW"]

    style CE fill:#e74c3c,stroke:#c0392b,color:#fff
    style CCE fill:#3498db,stroke:#2471a3,color:#fff
    style AUX fill:#e67e22,stroke:#ca6f1e,color:#fff
    style TOTAL fill:#8e44ad,stroke:#6c3483,color:#fff,stroke-width:2px
    style FWD fill:#2c3e50,stroke:#1a252f,color:#fff,stroke-width:2px
```

### Standalone SigLIP Training

To train only the vision encoder with contrastive learning:

```bash
python -m training.train_siglip
```

---

## Inference

### Interactive Generation

```bash
python -m inference.generate
```

Prompts for an image path and a text question, then generates a response autoregressively using top-p sampling with a paged KV-cache.

### Chat Interface

```bash
python -m inference.run_chat
```

Multi-turn chat with conversation memory, knowledge retrieval, and tool-augmented reasoning. Type `quit` to exit.

### Quick Demo

```bash
python -m demo
```

Runs a single inference pass on a test image and prints the result.

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 64 | Maximum tokens to generate |
| `temperature` | 0.8 | Sampling temperature (lower = more deterministic) |
| `top_p` | 0.9 | Nucleus sampling threshold |

### Speculative Decoding

The `SpeculativeDecoder` enables faster inference by running a small draft model ahead of the main model:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#2c3e50'}}}%%
sequenceDiagram
    participant D as Draft Model (small)
    participant T as Target Model (full VLM)
    participant O as Output Sequence

    rect rgb(235, 245, 251)
        Note over D,O: Repeat until EOS or max_tokens
        D->>D: Generate 4 candidate tokens (fast)
        D->>T: Send candidates for verification
        T->>T: Score all 4 in single forward pass
        T->>T: Accept where p_target / p_draft ≥ threshold
        T->>O: Emit accepted tokens (1-4 per step)
    end

    Note over D,O: Achieves near target-model quality at draft-model speed
```

---

## Evaluation & Benchmarks

### Run Full Evaluation

```bash
python -m evaluation.evaluate
```

### Run Benchmarks

```bash
python -m evaluation.run_benchmarks
```

### Metrics

| Task | Metric | Implementation |
|------|--------|----------------|
| **Image Captioning** | BLEU score | NLTK sentence-level BLEU with tokenized reference/hypothesis |
| **Visual QA** | Exact-match accuracy | Case-insensitive string comparison |
| **Image-Text Retrieval** | Recall@K | Cosine similarity ranking over normalized embeddings |

---

## Retrieval-Augmented Generation

The chat interface integrates a lightweight RAG pipeline that gives the model access to external knowledge without increasing parameters:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#2c3e50', 'fontSize': '13px'}}}%%
flowchart TB
    subgraph OFFLINE ["📦 Offline Indexing"]
        direction LR
        KB["knowledge/wiki.txt<br/>One passage per line"] --> LOAD["load_knowledge()"]
        LOAD --> EMB_D["SimpleEmbedder<br/>GPT-2 tokenize<br/>→ Embedding(50257, 768)<br/>→ Mean pool"]
        EMB_D --> INDEX["Document Index<br/>[N, 768] vectors"]
    end

    subgraph ONLINE ["🔍 Online Retrieval"]
        direction TB
        QUERY["User Question"] --> EMB_Q["Embed Query<br/>→ [1, 768]"]
        EMB_Q --> SIM["Cosine Similarity<br/>query vs all N documents"]
        INDEX --> SIM
        SIM --> TOPK["Select Top-k<br/>most relevant passages"]
    end

    subgraph GENERATE ["💬 Augmented Generation"]
        direction LR
        TOPK --> PROMPT["Build Prompt<br/>context + history + question"]
        PROMPT --> VLM["VLM Generate<br/>(image, augmented prompt)"]
        VLM --> ANS["Model Answer"]
    end

    style KB fill:#9b59b6,stroke:#7d3c98,color:#fff
    style INDEX fill:#f39c12,stroke:#ca6f1e,color:#fff
    style ANS fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
    style QUERY fill:#4a9eff,stroke:#2980b9,color:#fff
```

---

## Agent System

The project includes a ReAct-style (Reasoning + Acting) agent that decomposes complex queries into multi-step reasoning chains with tool calls:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#2c3e50', 'fontSize': '13px'}}}%%
flowchart TD
    START["User Query"] --> LOOP

    subgraph LOOP ["ReAct Loop (max 3 steps)"]
        direction TB
        THOUGHT["🧠 <b>Thought</b><br/>Reason about current state<br/>Determine next action"]
        ACTION["⚡ <b>Action</b><br/>ToolRouter selects tool<br/>based on keyword matching"]
        OBSERVE["👁️ <b>Observation</b><br/>Collect tool output<br/>Add to reasoning chain"]
        THOUGHT --> ACTION --> OBSERVE
        OBSERVE -->|"need more<br/>information"| THOUGHT
    end

    subgraph TOOLBOX ["🔧 Available Tools"]
        direction LR
        CALC["<b>Calculator</b><br/>Trigger: 'calculate'<br/>Evaluates math<br/>expressions"]
        CAPTION["<b>Captioner</b><br/>Trigger: 'describe image'<br/>Generates image<br/>descriptions"]
    end

    ACTION --> CALC & CAPTION

    OBSERVE -->|"reasoning<br/>complete"| FINAL["✅ <b>Final Answer</b>"]

    style THOUGHT fill:#3498db,stroke:#2471a3,color:#fff,stroke-width:2px
    style ACTION fill:#e67e22,stroke:#ca6f1e,color:#fff,stroke-width:2px
    style OBSERVE fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
    style FINAL fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
    style START fill:#4a9eff,stroke:#2980b9,color:#fff,stroke-width:2px
    style CALC fill:#fdf2e9,stroke:#e67e22,stroke-width:2px
    style CAPTION fill:#fdf2e9,stroke:#e67e22,stroke-width:2px
```

**Example interaction:**

```
User: "What is 224 * 224?"

Thought: The user wants to calculate a multiplication.
Action: calculator(224 * 224)
Observation: 50176
Thought: I have the answer.
Final Answer: 224 × 224 = 50,176
```

---

## Distributed Training

For multi-GPU training using Fully Sharded Data Parallel (FSDP):

```bash
python -m distributed.launch
```

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#2c3e50'}}}%%
flowchart TB
    LAUNCH["<b>launch.py</b><br/>mp.spawn(fn, world_size=N)"] --> INIT["Initialize Process Group<br/>nccl backend"]

    INIT --> G0 & G1 & GN

    subgraph FSDP_WRAP ["FSDP — Fully Sharded Data Parallel"]
        direction LR
        G0["<b>GPU 0</b><br/>Model Shard 1/N<br/>Optimizer Shard 1/N"]
        G1["<b>GPU 1</b><br/>Model Shard 2/N<br/>Optimizer Shard 2/N"]
        GN["<b>GPU N</b><br/>Model Shard N/N<br/>Optimizer Shard N/N"]

        G0 <-->|"All-Gather<br/>(forward)"| G1
        G1 <-->|"Reduce-Scatter<br/>(backward)"| GN
    end

    G0 & G1 & GN --> SYNC["Synchronized<br/>Optimizer Step"]
    SYNC --> SAVE["Checkpoint<br/>(rank 0 only)"]

    style LAUNCH fill:#2c3e50,stroke:#1a252f,color:#fff,stroke-width:2px
    style G0 fill:#76b900,stroke:#4a7a00,color:#fff,stroke-width:2px
    style G1 fill:#76b900,stroke:#4a7a00,color:#fff,stroke-width:2px
    style GN fill:#76b900,stroke:#4a7a00,color:#fff,stroke-width:2px
    style SAVE fill:#27ae60,stroke:#1e8449,color:#fff
```

FSDP shards model parameters, gradients, and optimizer states across GPUs, allowing training of models that don't fit on a single device.

---

## Configuration

All hyperparameters are managed through YAML configs in `configs/`:

### Model (`configs/model.yaml`)

```yaml
vision_dim: 768
text_dim: 768
num_layers: 12
num_heads: 8
ffn_dim: 3072
num_experts: 4
vocab_size: 50257
latent_tokens: 64
image_size: 224
patch_size: 16
```

### Training (`configs/training.yaml`)

```yaml
batch_size: 4
epochs: 50
max_tokens: 128
contrastive_weight: 0.0
moe_weight: 0.0
gradient_accumulation_steps: 1
```

### Optimizer (`configs/optimizer.yaml`)

```yaml
optimizer: adamw
lr: 5e-4
weight_decay: 0.01
betas: [0.9, 0.95]
```

---

## Pipeline

The full pipeline runs training through evaluation and inference in a single command:

```bash
./scripts/run_all.sh
```

Quick dry-run mode (caps training steps and skips interactive stages):

```bash
FAST_DRY_RUN=1 TRAIN_MAX_STEPS=20 ./scripts/run_all.sh
```

Wrapper validation mode (no real execution, useful for CI/local script checks):

```bash
SKIP_VENV_CHECK=1 SKIP_GPU_CHECK=1 SKIP_DATASET_CHECK=1 DRY_RUN_COMMANDS=1 FAST_DRY_RUN=1 ./scripts/run_all.sh
```

To enable interactive inference/chat stages explicitly:

```bash
RUN_INTERACTIVE=1 ./scripts/run_all.sh
```

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'lineColor': '#2c3e50'}}}%%
flowchart LR
    S1["<b>Step 1</b><br/>Activate<br/>venv"] --> S2["<b>Step 2</b><br/>GPU<br/>Check"]
    S2 --> S3["<b>Step 3</b><br/>Dataset<br/>Validation"]
    S3 --> S4["<b>Step 4</b><br/>Train VLM<br/>50 epochs"]
    S4 --> S5["<b>Step 5</b><br/>Evaluation<br/>Suite"]
    S5 --> S6["<b>Step 6</b><br/>Benchmarks<br/>BLEU·VQA·R@K"]
    S6 --> S7["<b>Step 7</b><br/>Demo<br/>Inference"]
    S7 --> S8["<b>Step 8</b><br/>Interactive<br/>Inference"]
    S8 --> S9["<b>Step 9</b><br/>Interactive<br/>Chat"]
    S9 --> DONE["Pipeline<br/>Complete ✓"]

    style S1 fill:#bdc3c7,stroke:#7f8c8d
    style S2 fill:#bdc3c7,stroke:#7f8c8d
    style S3 fill:#bdc3c7,stroke:#7f8c8d
    style S4 fill:#e74c3c,stroke:#c0392b,color:#fff,stroke-width:2px
    style S5 fill:#3498db,stroke:#2471a3,color:#fff
    style S6 fill:#3498db,stroke:#2471a3,color:#fff
    style S7 fill:#9b59b6,stroke:#7d3c98,color:#fff
    style S8 fill:#9b59b6,stroke:#7d3c98,color:#fff
    style S9 fill:#9b59b6,stroke:#7d3c98,color:#fff
    style DONE fill:#27ae60,stroke:#1e8449,color:#fff,stroke-width:2px
```

---

## Results

### Training Convergence

With the default configuration (50 epochs, batch size 4, learning rate 5e-4), the language modeling loss converges to near-zero on the instruction dataset:

```
step   0 | loss 11.2847
step  50 | loss  2.1433
step 100 | loss  0.3521
step 200 | loss  0.0012
step 300 | loss  0.0000
```

```mermaid
xychart-beta
    title "Training Loss Convergence"
    x-axis "Training Step" [0, 50, 100, 150, 200, 250, 300]
    y-axis "Loss" 0 --> 12
    line [11.28, 2.14, 0.35, 0.08, 0.001, 0.0, 0.0]
```

### Sample Outputs

| Image | Prompt | Model Output |
|-------|--------|-------------|
| ![dog](data/images/dog.jpg) | "What animal is this?" | "This is a dog." |
| ![car](data/images/car.jpg) | "What is shown in the image?" | "The image shows a car." |

---

## Key Design Decisions

- **SigLIP over CLIP** — SigLIP's sigmoid-based contrastive loss scales better than CLIP's softmax formulation for large batch sizes and avoids the need for a global normalization term.
- **Perceiver Resampler + Token Compressor** — Two-stage compression reduces 196 vision patches down to 32 tokens, cutting cross-attention cost by 6× without significant information loss.
- **GQA over MHA** — Grouped Query Attention reduces KV-cache memory by sharing key-value heads across query groups, critical for efficient autoregressive decoding.
- **MoE FFN** — Mixture-of-Experts with top-2 routing increases model capacity (4× more FFN parameters) while keeping per-token compute constant. The auxiliary load-balancing loss prevents expert collapse.
- **Weight-Tied LM Head** — The output projection shares weights with the input embedding, reducing parameter count by ~38M (768 × 50257) without hurting performance.
- **Paged KV-Cache** — Page-based memory management (16 tokens per page, 1024 max pages) prevents memory fragmentation during long autoregressive generations.
- **LoRA Support** — Low-Rank Adaptation (rank 8, alpha 16) enables parameter-efficient fine-tuning by injecting trainable low-rank matrices into frozen linear layers.

---

## License

This project is for research and educational purposes.

---

## Acknowledgments

This implementation draws on ideas from:

- [SigLIP](https://arxiv.org/abs/2303.15343) — Sigmoid loss for image-text pre-training
- [Flamingo](https://arxiv.org/abs/2204.14198) — Perceiver Resampler for vision-language models
- [Gemma](https://arxiv.org/abs/2403.08295) — Lightweight language model architecture
- [GQA](https://arxiv.org/abs/2305.13245) — Grouped Query Attention
- [Switch Transformers](https://arxiv.org/abs/2101.03961) — Mixture-of-Experts
- [LoRA](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation
- [ReAct](https://arxiv.org/abs/2210.03629) — Reasoning and Acting in language models
