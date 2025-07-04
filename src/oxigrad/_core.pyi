from typing import List, Optional, Union

class Value:
    """
    A scalar value in a computational graph used for automatic differentiation.

    This class represents a node in the computation graph, holding both its numerical
    value and the gradient of some final output with respect to this value. It supports
    various operations (addition, multiplication, etc.) and tracks how outputs depend
    on inputs for gradient calculation using backpropagation.
    """

    def __init__(self, data: Union[int, float], label: Optional[str] = None) -> None:
        """
        Initialize a new Value instance with the given data.

        Parameters:
            data: `Union[float, int]`: The numerical value of this node.
            label: `Optional[str]` (default `None`): label associated with this node.
        """

    @property
    def data(self) -> float:
        """
        Get the raw numerical value stored in this node.

        Returns:
            `float`: The scalar value this node holds.
        """

    @property
    def grad(self) -> float:
        """
        Get the gradient of the final output with respect to this node.

        Returns:
            `float`: The computed gradient after backpropagation.
        """

    @property
    def previous(self) -> List[Value]:
        """
        Get the list of parent nodes that were used to compute this Value.

        Returns:
            `List[Value]`: List of parent nodes
        """

    @property
    def operation(self) -> Optional[str]:
        """
        Get information about the operation used to compute this node.

        Returns:
            `Optional[str]`: Operation used to compute this node
        """

    def set_label(self, label: str) -> Value:
        """
        Assign a string label to this Value for visualization or debugging.

        Parameters:
            label: `str`: The label to assign.

        Returns:
            `Value`: The current Value instance with the label set.
        """

    def get_label(self) -> Optional[str]:
        """
        Retrieve the label assigned to this Value.

        Returns:
            `Optional[str]` - The label if it exists, otherwise `None`.
        """

    def pow(self, power: Union[int, float]) -> Value:
        """
        Raise this Value to the power of another Value.

        Parameters:
            power: `Value` - The exponent.

        Returns:
            `Value` - A new Value instance representing this ** power.
        """

    def backward(self) -> None:
        """
        Perform reverse-mode automatic differentiation.

        This function computes the gradient of the final output with respect to all
        nodes in the computation graph that lead to this Value. Should be called on
        a scalar output node to initiate backpropagation.
        """

    def __add__(self, other: Union[int, float, Value]) -> Value:
        """Add two values."""

    def __mul__(self, other: Union[int, float, Value]) -> Value:
        """Multiply two values."""

    def __pow__(self, power: Union[int, float]) -> Value:
        """Raise something to a power"""

    def __sub__(self, other: Union[int, float, Value]) -> Value:
        """Subtract two values."""

    def __neg__(self) -> Value:
        """Negate this value."""

    def __repr__(self) -> str:
        """String representation of this value."""

    def __str__(self) -> str:
        """String representation of this value."""

class Activation:
    """
    A collection of static methods implementing common activation functions.

    This class provides differentiable activation functions that operate on `Value`
    nodes within a computation graph. These functions return new `Value` instances
    with appropriate forward and backward computation logic for use in automatic
    differentiation.
    """

    @staticmethod
    def ReLU(input: Value) -> Value:
        """
        Apply the ReLU (Rectified Linear Unit) activation function.

        This function returns `input` if it is greater than 0, otherwise returns 0.

        Parameters:
            input: `Value`: The input Value to apply ReLU to.

        Returns:
            `Value`: A new Value instance after applying ReLU.

        Gradient:
            The gradient is 1.0 when input > 0, otherwise 0.
        """

    @staticmethod
    def Sigmoid(input: Value) -> Value:
        """
        Apply the Sigmoid activation function.

        This function maps the input to a value between 0 and 1 using the formula:
        sigmoid(x) = 1 / (1 + exp(-x))

        Parameters:
            input: `Value`: The input Value to apply sigmoid to.

        Returns:
            `Value`: A new Value instance after applying sigmoid.

        Gradient:
            Uses the identity: sigmoid(x) * (1 - sigmoid(x))
        """
