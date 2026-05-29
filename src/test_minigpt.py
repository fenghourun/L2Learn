import math
import pytest


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
# SHARED FIXTURES
# =========================================================

@pytest.fixture
def sample_matrix():
    from tensor import Tensor
    return Tensor([[1, 2], [3, 4]])


@pytest.fixture
def identity_matrix():
    from tensor import Tensor
    return Tensor([[1, 0], [0, 1]])


@pytest.fixture
def attention_module():
    from attention import SelfAttention
    return SelfAttention(d_model=4)


@pytest.fixture
def small_gpt():
    from transformer import GPT
    return GPT(vocab_size=10, d_model=8, n_layers=1)


# =========================================================
# TENSOR TESTS
# =========================================================

class TestTensorShape:

    @pytest.mark.parametrize(
        "data,expected",
        [
            ([], (0,)),
            ([[]], (1, 0)),
        ],
    )
    def test_empty_shapes(self, data, expected):
        from tensor import Tensor

        t = Tensor(data)
        assert t.shape() == expected


class TestTensorAddition:

    def test_scalar_broadcast_add(self, sample_matrix):
        out = sample_matrix + 10

        assert out.data == [
            [11, 12],
            [13, 14],
        ]

    def test_tensor_add(self, sample_matrix):
        from tensor import Tensor

        other = Tensor([[1, 2], [3, 4]])

        out = sample_matrix + other

        assert out.data == [
            [2, 4],
            [6, 8],
        ]

    def test_mismatched_shapes_fail(self):
        from tensor import Tensor

        a = Tensor([[1, 2]])
        b = Tensor([[1, 2], [3, 4]])

        with pytest.raises(Exception):
            _ = a + b


class TestTensorMatmul:

    def test_dimension_check(self):
        from tensor import Tensor

        a = Tensor([[1, 2, 3]])
        b = Tensor([[1, 2], [3, 4]])

        with pytest.raises(Exception):
            _ = a.matmul(b)

    def test_identity(self, sample_matrix, identity_matrix):
        out = sample_matrix.matmul(identity_matrix)

        assert out.data == sample_matrix.data


# =========================================================
# SOFTMAX TESTS
# =========================================================

class TestSoftmax:

    @pytest.mark.parametrize(
        "x",
        [
            [1000.0, 1001.0, 1002.0],
            [1000000.0, 1000001.0],
        ],
    )
    def test_numerical_stability(self, x):
        from attention import softmax

        y = softmax(x)

        assert approx(sum(y), 1.0)
        assert max(y) <= 1.0

    def test_single_value(self):
        from attention import softmax

        y = softmax([5.0])

        assert y == [1.0]

    def test_all_equal(self):
        from attention import softmax

        y = softmax([2.0, 2.0, 2.0])

        for v in y:
            assert approx(v, 1 / 3)


# =========================================================
# AUTOGRAD TESTS
# =========================================================

class TestAutograd:

    def test_zero_grad_accumulation(self):
        from autograd import Value

        a = Value(2.0)
        b = Value(3.0)

        c = a * b
        c.backward()

        grad1 = a.grad

        c.backward()

        # gradients accumulate unless reset
        assert a.grad >= grad1

    @pytest.mark.parametrize("depth", [1, 5, 10, 25])
    def test_long_chain(self, depth):
        from autograd import Value

        x = Value(1.0)

        y = x
        for _ in range(depth):
            y = y * Value(2.0)

        y.backward()

        assert x.grad > 0


# =========================================================
# ATTENTION TESTS
# =========================================================

class TestSelfAttention:

    def test_causal_mask_behavior(self, attention_module):
        x = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]

        out = attention_module.forward(x)

        assert len(out) == 2

    def test_constant_input_no_nan(self, attention_module):
        x = [[1, 1, 1, 1] for _ in range(5)]

        out = attention_module.forward(x)

        flat = flatten(out)

        for v in flat:
            assert not math.isnan(v)
            assert not math.isinf(v)

    def test_sequence_length_invariance(self, attention_module):
        x1 = [[1, 0, 0, 0]]

        x2 = [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]

        o1 = attention_module.forward(x1)
        o2 = attention_module.forward(x2)

        assert len(o1) == 1
        assert len(o2) == 2


# =========================================================
# NN LAYER TESTS
# =========================================================

class TestLinear:

    def test_forward_is_deterministic(self):
        from nn import Linear

        layer = Linear(2, 2)

        x = [1, 1]

        o1 = layer(x)
        o2 = layer(x)

        assert o1 == o2


class TestReLU:

    @pytest.mark.parametrize(
        "x",
        [
            [-5, -1, -0.1],
            [-999],
            [-1e-9],
        ],
    )
    def test_negative_only(self, x):
        from nn import ReLU

        r = ReLU()

        out = r(x)

        assert all(v == 0 for v in out)


# =========================================================
# DATASET TESTS
# =========================================================

class TestTextDataset:

    def test_empty_string_fails(self):
        from data import TextDataset

        ds = TextDataset("", block_size=5)

        with pytest.raises(Exception):
            _ = ds.get_batch()

    def test_single_char_dataset(self):
        from data import TextDataset

        ds = TextDataset("aaaaaa", block_size=3)

        x, y = ds.get_batch()

        assert len(x) == 3
        assert len(y) == 3

    def test_unknown_char_encode_fails(self):
        from data import TextDataset

        ds = TextDataset("abc", block_size=2)

        with pytest.raises(Exception):
            ds.encode("z")


# =========================================================
# GPT MODEL TESTS
# =========================================================

class TestGPT:

    def test_forward_empty_tokens_fails(self, small_gpt):
        with pytest.raises(Exception):
            _ = small_gpt.forward([])

    def test_output_shape_consistency(self, small_gpt):
        tokens = [1, 2, 3, 4, 5]

        out = small_gpt.forward(tokens)

        assert len(out) == len(tokens)


# =========================================================
# SAMPLING TESTS
# =========================================================

class TestSampling:

    def test_single_logit(self):
        from generate import sample

        logits = [10.0]

        for _ in range(5):
            assert sample(logits) == 0

    @pytest.mark.parametrize("temperature", [10.0, 100.0])
    def test_high_temperature(self, temperature):
        from generate import sample

        logits = [1.0, 2.0, 3.0]

        for _ in range(20):
            s = sample(logits, temperature=temperature)

            assert 0 <= s < len(logits)

    @pytest.mark.parametrize("temperature", [0.01, 0.0001])
    def test_low_temperature(self, temperature):
        from generate import sample

        logits = [1.0, 2.0, 3.0]

        s = sample(logits, temperature=temperature)

        assert 0 <= s < len(logits)