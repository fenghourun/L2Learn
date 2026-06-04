# Pure rust implementation of a mini GPT

## Goal

Build a minimal GPT implementation from scratch in Rust.

---

# 1. Tensors
# 2. Automatic differentiation

## Attention Helpers

```rust
tensor.masked_fill(mask, value) -> Tensor
```

---

# Layers

## Embedding

```rust
Embedding::new(vocab_size, embed_dim)
Embedding::forward(tokens)
```

Maps token ids → vectors.

## Linear

```rust
Linear::new(in_dim, out_dim)
Linear::forward(x)
```

Computes:

```text
y = xW + b
```

---

# Attention

## Single Head

```rust
Attention::new(d_model)
Attention::forward(x)
```

Computes:

```text
Q = XWq
K = XWk
V = XWv

scores = QKᵀ / sqrt(d)
weights = softmax(scores)

output = weightsV
```

## Causal Mask

Prevent tokens from attending to future tokens.

---

# Multi-Head Attention

```rust
MultiHeadAttention::new(
    d_model,
    num_heads
)
```

Split → attend → concat.

---

# LayerNorm

```rust
LayerNorm::new(dim)
LayerNorm::forward(x)
```

Computes:

```text
(x - mean) / sqrt(var + eps)
```

---

# MLP

```rust
MLP::new(
    d_model,
    hidden_dim
)
```

Computes:

```text
Linear
↓
GELU
↓
Linear
```

---

# Transformer Block

```rust
TransformerBlock::new(...)
TransformerBlock::forward(x)
```

Architecture:

```text
LayerNorm
↓
Attention
↓
Residual

LayerNorm
↓
MLP
↓
Residual
```

---

# GPT Model

```rust
GPT::new(...)
GPT::forward(tokens)
```

Components:

```text
Token Embedding
↓
Positional Embedding
↓
N Transformer Blocks
↓
Output Projection
```

---

# Generation

```rust
GPT::generate(prompt)
```

Loop:

```text
forward
↓
next token
↓
append
↓
repeat
```

Sampling strategies:

```rust
argmax()
temperature()
top_k()
```

---

# Training (Later)

## Loss

```rust
cross_entropy(logits, targets)
```

## Autograd

```rust
loss.backward()
```

## Optimizer

```rust
AdamW
```

Training loop:

```rust
for batch in dataset {
    let logits = model.forward(&x);

    let loss = cross_entropy(
        &logits,
        &targets,
    );

    loss.backward();

    optimizer.step();
    optimizer.zero_grad();
}
```

Run tests
```
cargo test
```
