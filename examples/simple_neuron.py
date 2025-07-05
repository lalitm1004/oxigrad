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
