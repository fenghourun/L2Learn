# autograd.py

class Value:
    """
    Scalar autograd engine (Micrograd-style).
    """

    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = data
        self.grad = 0.0

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

        self.requires_grad = requires_grad

    # -------------------------
    # ops
    # -------------------------

    def __add__(self, other):
        # TODO
        pass

    def __mul__(self, other):
        # TODO
        pass

    def __pow__(self, power):
        # TODO
        pass

    def relu(self):
        # TODO
        pass

    # -------------------------
    # backprop
    # -------------------------

    def backward(self):
        # TODO:
        # 1. topo sort graph
        # 2. set grad = 1 for output
        # 3. backprop in reverse order
        pass

