# Simple Neural Network in PyTorch

A basic implementation of a simple neural network using PyTorch, structured in an object-oriented programming (OOP) style. This network demonstrates the fundamentals of neural network computation, including forward propagation, activation functions, and weight matrices.

## Table of Contents

- [Overview](#overview)
- [Network Architecture](#network-architecture)
  - [Weight Matrices](#weight-matrices)
  - [Forward Pass Computation](#forward-pass-computation)
- [Activation Function](#activation-function)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Project](#running-the-project)
- [Example Output](#example-output)
- [Explanation of the Output](#explanation-of-the-output)
- [Customization](#customization)
  - [Changing Input Features](#changing-input-features)
  - [Modifying Weights](#modifying-weights)
  - [Changing the Activation Function](#changing-the-activation-function)
- [License](#license)

## Overview

This project implements a simple neural network from scratch using PyTorch tensors. The network is designed to process three input features and produce a binary output (0 or 1) based on a threshold activation function.

The neural network:

- Takes in three input features: `house`, `rock`, and `attr`.
- Processes the inputs through a single hidden layer with two neurons.
- Produces an output using one output neuron.
- Uses a custom activation function that applies a threshold at 0.5.

## Network Architecture

The network consists of:

- **Input Layer**: 3 neurons corresponding to the input features.
- **Hidden Layer**: 2 neurons.
- **Output Layer**: 1 neuron producing the final output.

### Weight Matrices

- **Weights between Input and Hidden Layer (`Wh`)**: A 2x3 matrix defining the weights from each input neuron to each hidden neuron.
- **Weights between Hidden and Output Layer (`Wout`)**: A vector of size 2 defining the weights from each hidden neuron to the output neuron.

### Forward Pass Computation

1. **Compute Hidden Layer Inputs (`Zh`)**:

   ```
   Zh = Wh x X
   ```

2. **Apply Activation Function to Hidden Layer Outputs (`Uh`)**:

   ```
   Uh = activation(Zh)
   ```

3. **Compute Output Neuron Input (`Zout`)**:

   ```
   Zout = Wout • Uh
   ```

4. **Apply Activation Function to Get Final Output (`Y`)**:

   ```
   Y = activation(Zout)
   ```

## Activation Function

A custom activation function is used:

```python
def activation(x):
    return 0 if x < 0.5 else 1
```

This function applies a simple threshold at 0.5. If the input `x` is less than 0.5, the output is `0`; otherwise, it's `1`.

## Project Structure

- `main.py`: The main script containing the `SimpleNeuralNetwork` class and execution code.
- `README.md`: Documentation file (this file).

## Getting Started

### Prerequisites

- Python 3.6 or higher
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/) (required by PyTorch)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/chimgan/simple-neural-network.git
   cd simple-neural-network
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install torch numpy
   ```

### Running the Project

Execute the `main.py` script:

```bash
python main.py
```

## Example Output

When you run the script, you should see output similar to:

```
Sum values at hidden layer neurons: tensor([0.3000, 1.4000])
Activated outputs from hidden layer neurons: tensor([0., 1.])
Sum value at output neuron before activation: 1.0
Final output after activation: 1
You are likely.
```

## Explanation of the Output

- **Sum values at hidden layer neurons**: The weighted sums (`Zh`) calculated for each hidden neuron.
- **Activated outputs from hidden layer neurons**: The outputs after applying the activation function to `Zh`.
- **Sum value at output neuron before activation**: The weighted sum (`Zout`) at the output neuron before activation.
- **Final output after activation**: The final output `Y` after applying the activation function to `Zout`.
- **Interpretation**: Based on the final output, a message is printed:
  - If `Y` is `1`, it prints `"You are likely."`
  - If `Y` is `0`, it prints `"Call me later."`

## Customization

You can modify the input features and weights to experiment with different scenarios.

### Changing Input Features

In `main.py`, adjust the values of `house`, `rock`, and `attr`:

```python
house = 1    # Feature 1
rock = 0     # Feature 2
attr = 1     # Feature 3
```

### Modifying Weights

Update the weight tensors in the `__init__` method of the `SimpleNeuralNetwork` class:

```python
self.Wh = torch.tensor([[0.3, 0.3, 0.0],
                        [0.4, -0.5, 1.0]], dtype=torch.float32)
self.Wout = torch.tensor([-1.0, 1.0], dtype=torch.float32)
```

### Changing the Activation Function

Replace the activation function with another, such as the sigmoid function:

```python
def activation(x):
    return 1 / (1 + torch.exp(-x))
```

Remember to adjust thresholding or interpretation logic accordingly.

## License

This project is open-source and available under the MIT License.

---

