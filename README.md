# Pure rust implementation of a mini GPT

## Goal

Build a minimal GPT implementation from scratch in Rust.

---

# Tensor

```rust
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}
```

## Construction

```rust
Tensor::new(data, shape) -> Tensor
Tensor::zeros(shape) -> Tensor
```

## Metadata

```rust
tensor.shape() -> &[usize]
tensor.numel() -> usize
```

## Shape Ops

```rust
tensor.reshape(shape) -> Tensor
tensor.transpose() -> Tensor
```

## Elementwise Ops

```rust
tensor.add(&other) -> Tensor
tensor.sub(&other) -> Tensor
tensor.mul(&other) -> Tensor
tensor.div(&other) -> Tensor
```

## Scalar Ops

```rust
tensor.add_scalar(x) -> Tensor
tensor.mul_scalar(x) -> Tensor
tensor.div_scalar(x) -> Tensor
```

## Matrix Ops

```rust
tensor.matmul(&other) -> Tensor
```

## Reductions

```rust
tensor.sum() -> f32
tensor.mean() -> f32
tensor.var() -> f32
```

## Activations

```rust
tensor.softmax() -> Tensor
tensor.gelu() -> Tensor
```

## Utilities

```rust
tensor.has_nan() -> bool
tensor.has_inf() -> bool
tensor.assert_finite()
```

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

---

# Build Order

1. Tensor
2. Matmul
3. Transpose
4. Softmax
5. Embedding
6. Linear
7. Single-Head Attention
8. Causal Mask
9. Multi-Head Attention
10. LayerNorm
11. MLP
12. Transformer Block
13. GPT
14. Text Generation
15. Autograd
16. AdamW
17. Training



Run tests
```
cargo test
```
