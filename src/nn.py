# nn.py

import random


class Linear:
    def __init__(self, in_dim, out_dim):
        # TODO: initialize weights + bias
        self.W = None
        self.b = None

    def __call__(self, x):
        # TODO: xW + b
        pass


class ReLU:
    def __call__(self, x):
        # TODO
        pass


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = None
        self.beta = None

    def __call__(self, x):
        # TODO: normalize vector
        pass
