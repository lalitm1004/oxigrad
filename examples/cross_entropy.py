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
