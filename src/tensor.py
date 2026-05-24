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


    def __add__(self, other):
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

        # Case 1: other is a number
        # return elementwise addition
        if isinstance(other, int):
            pass

        # Case 2: other is tensor
        # elementwise addition

    def recurse(self):
        curr_data = self.data
        for isinstance(curr_data, list):
            for i in range(dimension):
                curr_element = curr_data[i]
            curr_data = curr_data[0]
    
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
        pass

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
