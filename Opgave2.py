import numpy as np
import matplotlib.pyplot as plt

# Generate the input data
X = np.linspace(-1, 1, 50).reshape(50, 1)
y = np.sin(X)

# Define the hyperparameters
eta = 0.5
num_hidden = 3
num_input = 1
num_output = 1

# Initialize the weights randomly
W1 = np.random.randn(num_input, num_hidden)
W2 = np.random.randn(num_hidden, num_output)

# Define the activation function
def tanh(x):
    return np.tanh(x)

# Define the derivative of the activation function
def tanh_prime(x):
    return 1 - np.tanh(x)**2

# Train the network using backpropagation
for i in range(131):
    # Forward pass
    hidden = tanh(np.dot(X, W1))
    output = np.dot(hidden, W2)

    # Compute the error
    error = y - output

    # Backpropagate the error
    delta_output = error
    delta_hidden = np.dot(delta_output, W2.T) * tanh_prime(hidden)

    # Update the weights
    W2 += eta * np.dot(hidden.T, delta_output)
    W1 += eta * np.dot(X.T, delta_hidden)

# Predict the output using the trained network
hidden = tanh(np.dot(X, W1))
predicted_output = np.dot(hidden, W2)

# Plot the predicted function
#plt.plot(X, y, label='ground truth')
plt.plot(predicted_output, label='predicted')
plt.legend()
plt.show()