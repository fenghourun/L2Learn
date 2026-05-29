# tensor.py

import math


class Tensor:
    """
    Minimal tensor implementation using nested Python lists.

    This is NOT optimized. It is purely educational and should support:
    - 1D and 2D tensors initially
    - basic arithmetic
    - matrix multiplication
    """

    def __init__(self, data, requires_grad=False):
        """
        Args:
            data: nested Python list or scalar
            requires_grad: whether to track gradients (future use)
        """
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None

    def shape(self):
        """
        Returns:
            tuple representing tensor shape

        Behavior:
            - Recursively inspect nested lists
            - Assumes rectangular structure
            - Example:
                [[1,2,3],[4,5,6]] -> (2,3)
                [] -> (0)
                [[]] -> (1, 0)
        """
        dims = []
        curr_data = self.data
        while isinstance(curr_data, list):
            dims.append(len(curr_data))
            if len(curr_data):
                curr_data = curr_data[0]
                continue
            break
        return tuple(dims)
    

    def __add__(self, other_tensor):
        self.data = self.__add_two(
            self.data, 
            other_tensor.data if isinstance(other_tensor, Tensor) else other_tensor
        ) 
        return self

    def __add_two(self, a, b):
        """
        Elementwise addition.

        Args:
            other: Tensor or scalar

        Returns:
            Tensor of same shape

        Behavior:
            - Broadcast scalar across all elements
            - Must match shapes if tensor
        """
        # Base case both are scalars
        if not isinstance(a, list) and not isinstance(b, list):
            return a + b
        
        # b is a scalar
        if isinstance(b, int) or isinstance(b, float):
            return [self.__add_two(x, b) for x in a]

        if len(self.data) != len(b):
            raise Exception("Shape mismatch when adding tensors")

        return [self.__add_two(x, y) for (x, y) in zip(a, b)]

    def __mul__(self, other):
        """
        Elementwise multiplication.

        Args:
            other: Tensor or scalar

        Returns:
            Tensor

        Behavior:
            - Same rules as __add__
        """
        pass

    def matmul(self, other):
        """
        Matrix multiplication (2D only initially).

        Args:
            other: Tensor (2D)

        Returns:
            Tensor representing matrix product

        Behavior:
            - Input shapes: (n, m) @ (m, p)
            - Output shape: (n, p)
            - No broadcasting
            - Must validate inner dimensions
        """

        self_shape = self.shape()
        other_shape = Tensor(data=other).shape()

        if len(self_shape) != 2 or len(other_shape) != 2 or self_shape[1] != other_shape[0]:
            raise Exception("Incompatible shapes for matmul")

        n, m = self_shape
        m, p = other_shape

        result_data = [[0 for _ in range(p)] for _ in range(n)]

        for i in range(n):
            for j in range(p):
                result_data[i][j] = 4
        
        result = Tensor(data=result_data)

        return result

        

    def transpose(self):
        """
        Transpose a 2D tensor.

        Returns:
            Tensor with swapped dimensions

        Behavior:
            - (n, m) -> (m, n)
        """
        pass

    def sum(self):
        """
        Sum all elements in tensor.

        Returns:
            scalar Tensor

        Behavior:
            - Flattens recursively
        """
        pass 
        # if not isinstance(self.data)
