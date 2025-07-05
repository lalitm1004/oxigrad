from typing import List, Optional, Sequence, Union

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
        Initialize a new Value instance.

        Parameters:
            data: `Union[int, float]` - The numerical value of this node.
            label: `Optional[str]` (default `None`) - Label associated with this node.
        """
        ...

    @property
    def data(self) -> float:
        """
        Get the raw numerical value stored in this node.

        Returns:
            `float`
        """
        ...
    
    @data.setter
    def data(self, new_data: Union[int, float]) -> None: ...

    @property
    def grad(self) -> float:
        """
        Get the gradient of the final output with respect to this node.

        Returns:
            `float`
        """
        ...

    @property
    def previous(self) -> List[Value]:
        """
        Get the list of parent nodes used to compute this Value.

        Returns:
            `List[Value]`
        """
        ...

    @property
    def operation(self) -> Optional[str]:
        """
        Get information about the operation used to compute this node.

        Returns:
            `Optional[str]`
        """
        ...

    def set_label(self, label: str) -> Value:
        """
        Assign a string label for visualization or debugging.

        Parameters:
            label: `str` - The label to assign.

        Returns:
            `Value` - The current Value instance with the label set.
        """
        ...

    def get_label(self) -> Optional[str]:
        """
        Retrieve the label assigned to this Value.

        Returns:
            `Optional[str]`
        """
        ...

    def exp(self) -> Value:
        """
        Return e^self.

        Returns:
            `Value`
        """
        ...

    def backward(self) -> None:
        """
        Perform reverse-mode automatic differentiation.

        Computes the gradient of the final output with respect to all
        nodes in the computation graph that lead to this Value. Should be
        called on a scalar output node to initiate backpropagation.
        """
        ...

    # Standard arithmetic operations
    def __add__(self, other: Union[int, float, Value]) -> Value: ...
    def __mul__(self, other: Union[int, float, Value]) -> Value: ...
    def __pow__(self, power: Union[int, float]) -> Value: ...
    def __sub__(self, other: Union[int, float, Value]) -> Value: ...
    def __truediv__(self, other: Union[int, float, Value]) -> Value: ...
    def __neg__(self) -> Value: ...

    # Reverse artihmetic operations
    def __radd__(self, other: Union[int, float, Value]) -> Value: ...
    def __rmul__(self, other: Union[int, float, Value]) -> Value: ...
    def __rsub__(self, other: Union[int, float, Value]) -> Value: ...
    def __rtruediv__(self, other: Union[int, float, Value]) -> Value: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Activation:
    """
    Collection of static methods implementing common activation functions.

    Provides activation functions that operate on `Value`
    nodes within a computation graph.
    """

    @staticmethod
    def ReLU(input: Value) -> Value:
        """
        Apply the ReLU (Rectified Linear Unit) activation function.

        Returns `input` if it is greater than 0, otherwise returns 0.

        Parameters:
            input: `Value` - Input Value to apply ReLU to.

        Returns:
            `Value`

        Gradient:
            The gradient is 1.0 when input > 0, otherwise 0.
        """
        ...

    @staticmethod
    def Sigmoid(input: Value) -> Value:
        """
        Apply the Sigmoid activation function.

        Maps the input to a value between 0 and 1 using:
        sigmoid(x) = 1 / (1 + exp(-x))

        Parameters:
            input: `Value` - Input Value to apply sigmoid to.

        Returns:
            `Value`

        Gradient:
            Uses the identity: sigmoid(x) * (1 - sigmoid(x))
        """
        ...

    @staticmethod
    def Softmax(logits: List[Value]) -> Sequence[float]:
        """
        Apply the Softmax function to a list of logits.

        Converts logits to probabilities that sum to 1.

        **NOTE: CANNOT BE USED TO CALCULATE GRADIENT VALUES**

        Parameters:
            logits: `List[Value]` - Input logits to transform.

        Returns:
            `Sequence[float]` - Probability distribution over inputs.
        """
        ...

class Loss:
    """Collection of static methods implementing loss functions."""

    @staticmethod
    def CrossEntropy(logits: Sequence[Value], targets: Sequence[Value]) -> Value:
        """
        Compute cross-entropy loss between logits and targets. Has in-built Softmax

        Parameters:
            logits: `Sequence[Value]` - Predicted output logits.
            targets: `Sequence[Value]` - Ground truth target values.

        Returns:
            `Value` - Computed cross-entropy loss.
        """
        ...
