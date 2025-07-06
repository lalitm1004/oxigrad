from oxigrad import Loss, Value
from matplotlib import pyplot as plt
import random

ITERATIONS = 1000
LR = 0.002

x_train = [Value(4.0), Value(7.0), Value(9.0), Value(11.0)]
y_train = [Value(6.5), Value(10.1), Value(11.8), Value(14.2)]

# Linear Regression: we optimize for y = mx + c
# For simple linear regression, we need only one weight (m) and one bias (c)
m = Value(random.random())  # slope
c = Value(random.random())  # intercept

losses = []  # to track loss over iterations

# training loop
for iteration in range(ITERATIONS):
    y_list = []
    
    # Forward pass: compute predictions
    for x0 in x_train:
        y0 = (m * x0) + c
        y_list.append(y0)
    
    # Compute loss
    loss = Loss.MSE(y_list, y_train)
    losses.append(loss.data)  # store loss for plotting
    
    # Backward pass
    loss.backward()
    
    # Update parameters using gradient descent
    m.data -= LR * m.grad
    c.data -= LR * c.grad
    
    # Zero gradients for next iteration
    m.zero_grad()
    c.zero_grad()
        
    # Print progress every 100 iterations
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss.data:.4f}")

print(f"\nFinal parameters:")
print(f"m (slope): {m.data:.4f}")
print(f"c (intercept): {c.data:.4f}")

# Plot results
plt.figure(figsize=(12, 4))

# Plot 1: Training data and fitted line
plt.subplot(1, 2, 1)
x_vals = [x.data for x in x_train]
y_vals = [y.data for y in y_train]
plt.scatter(x_vals, y_vals, color='blue', label='Training data')

# Plot the fitted line
x_range = [min(x_vals) - 1, max(x_vals) + 1]
y_range = [m.data * x + c.data for x in x_range]
plt.plot(x_range, y_range, 'r-', label=f'y = {m.data:.2f}x + {c.data:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)

# Plot 2: Loss over iterations
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.tight_layout()
plt.show()

# Make predictions on new data
test_x = [Value(5.0), Value(8.0), Value(12.0)]
print(f"\nPredictions:")
for x in test_x:
    prediction = m * x + c
    print(f"x = {x.data}, predicted y = {prediction.data:.2f}")