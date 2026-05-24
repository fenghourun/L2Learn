# attention.py

import math


def softmax(xs):
    """
    Numerically stable softmax.

    Args:
        xs: list of floats

    Returns:
        probability distribution (sum = 1)

    Behavior:
        - subtract max(x) for stability
        - exponentiate values
        - normalize by sum
    """
    pass


class SelfAttention:
    """
    Single-head causal self-attention.
    """

    def __init__(self, d_model):
        """
        Args:
            d_model: embedding dimension

        Behavior:
            - initialize Q, K, V projection matrices
            - initialize output projection matrix
        """
        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.Wo = None

    def forward(self, x):
        """
        Args:
            x: sequence of token embeddings
               shape: (T, d_model)

        Returns:
            transformed sequence (T, d_model)

        Steps:
            1. Compute Q = xWq, K = xWk, V = xWv
            2. Compute attention scores: QK^T
            3. Scale by sqrt(d_model)
            4. Apply causal mask (future tokens blocked)
            5. Apply softmax over last dimension
            6. Multiply weights with V
            7. Apply output projection Wo
        """
        pass
