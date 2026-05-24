# tensor.py

import math


class Tensor:
    """
    Minimal tensor backed by nested Python lists.
    Future upgrades: autograd, broadcasting, GPU (not yet).
    """

    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None

        # for autograd later
        self._backward = lambda: None
        self._prev = set()

    # -------------------------
    # Basic utilities
    # -------------------------

    def shape(self):
        # TODO: compute recursive shape of nested lists
        pass

    def zeros_like(self):
        # TODO
        pass

    def __repr__(self):
        return f"Tensor({self.data})"

    # -------------------------
    # Elementwise ops
    # -------------------------

    def __add__(self, other):
        # TODO: elementwise addition
        pass

    def __mul__(self, other):
        # TODO: elementwise multiplication
        pass

    # -------------------------
    # Matrix operations
    # -------------------------

    def matmul(self, other):
        """
        TODO:
        implement matrix multiplication for 2D lists
        """
        pass

    def transpose(self):
        # TODO
        pass

    # -------------------------
    # Reductions
    # -------------------------

    def sum(self):
        # TODO
        pass
