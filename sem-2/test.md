# Introduction to Neural Networks
**Date:** 2026-02-11
**Subject:** AI/ML

## 1. The Perceptron
A perceptron is the fundamental unit of a neural network. It takes inputs, applies weights, and passes the sum through an activation function.

### Mathematical Model
The output $y$ is calculated as:
$$y = f(\sum_{i=1}^{n} w_i x_i + b)$$

Where:
* $w$ = weights
* $x$ = inputs
* $b$ = bias
* $f$ = activation function (e.g., Sigmoid, ReLU)

## 2. Implementation in Python
Here is a simple neuron implementation using NumPy:

```python
import numpy as np

def sigmoid(x):
    # Activation function
    return 1 / (1 + np.exp(-x))

def neuron(inputs, weights, bias):
    total = np.dot(inputs, weights) + bias
    return sigmoid(total)

inputs = np.array([2, 3])
weights = np.array([0.5, 0.9])
bias = 0.1

print(f"Output: {neuron(inputs, weights, bias)}")