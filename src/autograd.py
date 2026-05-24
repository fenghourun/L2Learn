# autograd.py


class Value:
    """
    Scalar autograd engine (Micrograd-style).

    Each Value represents a single scalar and tracks:
    - data (float)
    - gradient
    - computation graph dependencies
    """

    def __init__(self, data, _children=(), _op=""):
        """
        Args:
            data: float scalar value
            _children: parent nodes in computation graph
            _op: operation that created this node
        """
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        """
        Scalar addition.

        Behavior:
            - returns new Value
            - builds computation graph
        """
        pass

    def __mul__(self, other):
        """
        Scalar multiplication.

        Behavior:
            - returns new Value
            - tracks graph dependencies
        """
        pass

    def backward(self):
        """
        Backpropagation entry point.

        Expected behavior:
            1. Build topological ordering of computation graph
            2. Initialize output gradient to 1.0
            3. Propagate gradients backward using chain rule
            4. Accumulate gradients in each node

        Constraints:
            - Must handle DAG correctly
            - Must support repeated nodes (accumulation)
        """
        pass
