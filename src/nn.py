# nn.py

import random


class Linear:
    """
    Fully connected layer: y = xW + b
    """

    def __init__(self, in_dim, out_dim):
        """
        Args:
            in_dim: input feature size
            out_dim: output feature size

        Behavior:
            - Initialize weights with small random values
            - Initialize bias to zeros or small random values
            - Store parameters as Python lists (no NumPy)
        """
        self.W = None
        self.b = None

    def __call__(self, x):
        """
        Forward pass.

        Args:
            x: 1D or 2D list (vector or batch)

        Returns:
            transformed output vector

        Behavior:
            - compute xW + b
            - must support single vector first
        """
        pass


class ReLU:
    """
    ReLU activation function.
    """

    def __call__(self, x):
        """
        Args:
            x: scalar or list

        Returns:
            max(0, x) applied elementwise

        Behavior:
            - negative values become 0
            - positive values unchanged
        """
        pass


class LayerNorm:
    """
    Normalizes features across hidden dimension.
    """

    def __init__(self, dim, eps=1e-5):
        """
        Args:
            dim: feature dimension
            eps: numerical stability constant
        """
        self.gamma = None
        self.beta = None
        self.eps = eps

    def __call__(self, x):
        """
        Args:
            x: vector (list of floats)

        Returns:
            normalized vector

        Behavior:
            1. compute mean of x
            2. compute variance
            3. normalize: (x - mean) / sqrt(var + eps)
            4. apply scale (gamma) and shift (beta)
        """
        pass
