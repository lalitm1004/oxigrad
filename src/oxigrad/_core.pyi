from typing import Optional, Union

class Value:
    """
    A scalar value in a computational graph used for automatic differentiation.

    This class represents a node in the computation graph, holding both its numerical
    value and the gradient of some final output with respect to this value. It supports
    various operations (addition, multiplication, etc.) and tracks how outputs depend
    on inputs for gradient calculation using backpropagation.
    """

    def __init__(self, data: Union[int, float]) -> None:
        """
        Initialize a new Value instance with the given data.

        Parameters:
            data: `Union[float, int]`: The numerical value of this node.
        """

    @staticmethod
    def from_float(data: float) -> Value:
        """
        Create a Value object from a float.

        Parameters:
            data: `float`: The numeric value to be wrapped.

        Returns:
            `Value`: A Value instance containing the given float.
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

    def pow(self, power: Value) -> Value:
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

    def __add__(self, other: Value) -> Value:
        """Add two values."""

    def __mul__(self, other: Value) -> Value:
        """Multiply two values."""

    def __sub__(self, other: Value) -> Value:
        """Subtract two values."""

    def __neg__(self) -> Value:
        """Negate this value."""

    def __repr__(self) -> str:
        """String representation of this value."""

    def __str__(self) -> str:
        """String representation of this value."""
