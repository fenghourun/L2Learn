# attention.py

import math


def softmax(xs):
    # TODO: numerically stable softmax
    pass


class SelfAttention:
    def __init__(self, d_model):
        self.d_model = d_model

        # TODO: Q, K, V weights
        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.Wo = None

    def forward(self, x):
        """
        x: sequence of vectors (T x d_model)
        """

        # TODO:
        # 1. compute Q, K, V
        # 2. attention scores = QK^T / sqrt(d)
        # 3. apply causal mask
        # 4. softmax
        # 5. weighted sum over V
        # 6. output projection

        pass
