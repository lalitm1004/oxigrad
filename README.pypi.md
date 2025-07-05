[![PyPI version info](https://img.shields.io/pypi/v/oxigrad.svg?style=for-the-badge&logo=pypi&color=yellowgreen&logoColor=white)](https://pypi.python.org/pypi/oxigrad)
[![PyPI supported Python versions](https://img.shields.io/pypi/pyversions/oxigrad.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.python.org/pypi/oxigrad)

## Key Features
- Forward and Backward pass on scalar values
- Backpropagation with a computation graph
- Activation Functions:
    - Sigmoid
    - ReLU
    - Softmax
- Loss Criterion:
    - MSE
    - Cross Entropy

## Installing
```bash
# using pip
pip install oxigrad

# using uv
uv add oxigrad
```
> *Note: Python 3.9 or higher is required*

## Quick Examples

### Simple Neuron
```python
from oxigrad import Value, Activation

x1 = Value(1.7, label="x1")
x2 = Value(-0.3, label="x2")

w1 = Value(-1.5, label="w1")
w2 = Value(0.1, label="w2")

b = Value(0.5, label="b")

# Set a label after an operation
x1w1 = (x1 * w1).set_label("x1w1")
x2w2 = x2 * w2

xwb = x1w1 + x2w2 + b

z = Activation.Sigmoid(xwb).set_label("z")

# Run backpropagation
z.backward()

print(z)    # Value(data=0.8889, grad=1.0000, label='z', operation='SIGMOID')
print(w1)   # Value(data=-1.5000, grad=0.1678, label='w1')
print(w2)   # Value(data=0.1000, grad=-0.0296, label='w2')
print(b)    # Value(data=0.5000, grad=0.0987, label='b')
```

### Cross Entropy Loss
```python
from oxigrad import Activation, Loss, Value

a = Value(1.63, label="logit_0")
b = Value(0.27, label="logit_1")

c = Value(1, label="target_0")
d = Value(0, label="target_1")

logits = [a, b]
targets = [c, d]

# Convert logits into probability scores (does not add to computation graph)
probability_scores = Activation.Softmax(logits)
print(probability_scores)       # [0.7957596977159083, 0.20424030228409182]
print(sum(probability_scores))  # 1.0

# Has build in softmax
loss = Loss.CrossEntropy(logits, targets).set_label("loss")
loss.backward()

print(loss) # Value(data=0.2285, grad=1.0000, label='loss', operation='CROSSENTROPY')
print(a)    # Value(data=1.6300, grad=-0.2042, label='logit_0')
print(b)    # Value(data=0.2700, grad=0.2042, label='logit_1')
print(c)    # Value(data=1.0000, grad=0.2285, label='target_0')
print(d)    # Value(data=0.0000, grad=1.5885, label='target_1')
```