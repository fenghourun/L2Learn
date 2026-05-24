import pytest
import math
import random


# =========================================================
# HELPERS
# =========================================================

def approx(a, b, eps=1e-5):
    return abs(a - b) < eps


def flatten(x):
    if isinstance(x, list):
        return [i for sub in x for i in flatten(sub)]
    return [x]


# =========================================================
# 1. TENSOR EDGE CASES
# =========================================================

def test_tensor_empty_shape():
    from tensor import Tensor

    t = Tensor([])
    assert t.shape() == (0,)

def test_tensor_empty_shape_2():
    from tensor import Tensor

    t = Tensor([[]])
    assert t.shape() == (1,0)

def test_tensor_scalar_broadcast_add():
    from tensor import Tensor

    t = Tensor([[1, 2], [3, 4]])
    out = t + 10

    assert out.data == [[11, 12], [13, 14]]


def test_tensor_mismatched_add_fails():
    from tensor import Tensor

    a = Tensor([[1, 2]])
    b = Tensor([[1, 2], [3, 4]])

    with pytest.raises(Exception):
        _ = a + b


def test_tensor_matmul_dimension_check():
    from tensor import Tensor

    a = Tensor([[1, 2, 3]])
    b = Tensor([[1, 2], [3, 4]])

    # invalid inner dimensions -> must fail
    with pytest.raises(Exception):
        _ = a.matmul(b)


def test_tensor_matmul_identity():
    from tensor import Tensor

    a = Tensor([[1, 2], [3, 4]])
    I = Tensor([[1, 0], [0, 1]])

    out = a.matmul(I)
    assert out.data == a.data


# =========================================================
# 2. SOFTMAX EDGE CASES
# =========================================================

def test_softmax_numerical_stability_large_values():
    from attention import softmax

    x = [1000.0, 1001.0, 1002.0]

    y = softmax(x)

    assert approx(sum(y), 1.0)
    assert max(y) <= 1.0


def test_softmax_single_value():
    from attention import softmax

    x = [5.0]

    y = softmax(x)

    assert y == [1.0]


def test_softmax_all_equal():
    from attention import softmax

    x = [2.0, 2.0, 2.0]

    y = softmax(x)

    for v in y:
        assert approx(v, 1/3)


# =========================================================
# 3. AUTOGRAD EDGE CASES
# =========================================================

def test_autograd_zero_grad_accumulation():
    from autograd import Value

    a = Value(2.0)
    b = Value(3.0)

    c = a * b
    c.backward()

    grad1 = a.grad

    c.backward()

    # gradients should accumulate unless reset
    assert a.grad >= grad1


def test_autograd_chain_long():
    from autograd import Value

    x = Value(1.0)

    y = x
    for _ in range(10):
        y = y * Value(2.0)

    y.backward()

    assert x.grad > 0


# =========================================================
# 4. ATTENTION EDGE CASES
# =========================================================

def test_attention_causal_mask_behavior():
    from attention import SelfAttention

    attn = SelfAttention(d_model=4)

    x = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ]

    out = attn.forward(x)

    assert len(out) == 2


def test_attention_constant_input():
    from attention import SelfAttention

    attn = SelfAttention(d_model=4)

    x = [[1, 1, 1, 1] for _ in range(5)]

    out = attn.forward(x)

    # output should not be NaN or inf
    flat = flatten(out)

    for v in flat:
        assert not math.isnan(v)
        assert not math.isinf(v)


def test_attention_sequence_length_invariance():
    from attention import SelfAttention

    attn = SelfAttention(d_model=4)

    x1 = [[1, 0, 0, 0]]
    x2 = [[1, 0, 0, 0], [1, 0, 0, 0]]

    o1 = attn.forward(x1)
    o2 = attn.forward(x2)

    assert len(o1) == 1
    assert len(o2) == 2


# =========================================================
# 5. NN LAYER EDGE CASES
# =========================================================

def test_linear_deterministic_forward():
    from nn import Linear

    layer = Linear(2, 2)

    x = [1, 1]

    o1 = layer(x)
    o2 = layer(x)

    assert o1 == o2


def test_relu_negative_only():
    from nn import ReLU

    r = ReLU()

    x = [-5, -1, -0.1]

    out = r(x)

    assert all(v == 0 for v in out)


# =========================================================
# 6. DATASET EDGE CASES
# =========================================================

def test_dataset_empty_string():
    from data import TextDataset

    ds = TextDataset("", block_size=5)

    with pytest.raises(Exception):
        _ = ds.get_batch()


def test_dataset_single_char():
    from data import TextDataset

    ds = TextDataset("aaaaaa", block_size=3)

    x, y = ds.get_batch()

    assert len(x) == 3
    assert len(y) == 3


def test_dataset_encode_unknown_char():
    from data import TextDataset

    ds = TextDataset("abc", block_size=2)

    with pytest.raises(Exception):
        ds.encode("z")


# =========================================================
# 7. MODEL EDGE CASES
# =========================================================

def test_model_forward_empty_tokens():
    from transformer import GPT

    model = GPT(vocab_size=10, d_model=8, n_layers=1)

    with pytest.raises(Exception):
        _ = model.forward([])


def test_model_output_shape_consistency():
    from transformer import GPT

    model = GPT(vocab_size=10, d_model=8, n_layers=1)

    tokens = [1, 2, 3, 4, 5]

    out = model.forward(tokens)

    assert len(out) == len(tokens)


# =========================================================
# 8. SAMPLING EDGE CASES
# =========================================================

def test_sampling_single_logit():
    from generate import sample

    logits = [10.0]

    for _ in range(5):
        assert sample(logits) == 0


def test_sampling_temperature_extreme():
    from generate import sample

    logits = [1.0, 2.0, 3.0]

    # very high temperature → randomness allowed but valid range
    for _ in range(20):
        s = sample(logits, temperature=10.0)
        assert 0 <= s < len(logits)

    # very low temperature → near deterministic (no crash)
    s = sample(logits, temperature=0.01)
    assert 0 <= s < len(logits)
