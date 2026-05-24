
# MiniGPT — Pure Python Transformer Implementation Guide

This project is a bottom-up implementation of a GPT-style language model in pure Python. No ML frameworks are used. The goal is full understanding of how transformers work from first principles.

---

# 🧭 Goal

Build a working autoregressive language model that can:
- learn from raw text
- predict next tokens
- generate coherent sequences

All implemented from scratch in pure Python.

--- 

# 📦 Module Guide

## 1. `tensor.py` — Core Data Structure

### Purpose
Implements a minimal tensor system using nested Python lists.

### You will implement:
- matrix multiplication
- elementwise operations (+, *)
- transpose
- sum/reductions

### Key idea
Tensors are just structured arrays — no magic.

---

## 2. `autograd.py` — Gradient Engine (optional but recommended)

### Purpose
Implements scalar-level automatic differentiation.

### You will implement:
- computational graph
- backward pass
- gradient tracking for scalars

### Key idea
Learning = chaining derivatives through a graph.

---

## 3. `nn.py` — Neural Network Primitives

### Purpose
Basic building blocks of neural networks.

### You will implement:
- Linear layer (xW + b)
- activation functions (ReLU, GELU)
- LayerNorm

### Key idea
Neural nets are compositions of simple functions.

---

## 4. `attention.py` — Self-Attention (core component)

### Purpose
Implements transformer attention from scratch.

### You will implement:
- Q, K, V projections
- scaled dot-product attention
- causal masking
- softmax normalization

### Key idea
Attention = weighted information retrieval across tokens.

---

## 5. `transformer.py` — Full Model

### Purpose
Stacks attention + MLP into transformer blocks.

### You will implement:
- transformer block (attention + MLP + residuals)
- GPT model class
- token + positional embeddings

### Key idea
Transformers are repeated computation blocks.

---

## 6. `data.py` — Dataset Pipeline

### Purpose
Loads and prepares text data.

### You will implement:
- vocabulary (start char-level)
- encode/decode functions
- batch generation

### Key idea
Language modeling = next-token prediction on sequences.

---

## 7. `train.py` — Training Loop

### Purpose
Trains the model end-to-end.

### You will implement:
- forward pass
- loss computation (cross entropy)
- backpropagation
- optimizer (start with SGD)

### Key idea
Training is just repeated prediction + error correction.

---

## 8. `generate.py` — Text Generation

### Purpose
Autoregressive sampling from trained model.

### You will implement:
- softmax sampling
- temperature control
- token-by-token generation loop

### Key idea
Inference = repeated next-token sampling.

---

# 🚀 Implementation Order (IMPORTANT)

Follow strictly:

1. tensor.py (matrix ops first)
2. attention.py (manual math version)
3. transformer.py (forward pass only)
4. data.py (char tokenizer first)
5. train.py (even if model is dumb initially)
6. fix learning + stability
7. generate.py

---

# 🧠 Core Learning Principles

- Do not use external ML libraries
- Do not optimize early
- Prefer correctness over performance
- Build the simplest working version first
- Upgrade components incrementally

---

# 🎯 Milestones

## Milestone 1
Tensor operations + attention implemented

## Milestone 2
Forward pass of transformer works

## Milestone 3
Model can overfit tiny dataset

## Milestone 4
Text generation works (even if poor quality)

---

# ⚠️ Common Mistakes

- Trying to implement everything at once
- Using PyTorch/HuggingFace too early
- Skipping attention math understanding
- Overengineering before first working model

---

# 🧭 Final Outcome

A fully working miniature GPT-style model built entirely from scratch, where every component is understood and implemented manually.

---
