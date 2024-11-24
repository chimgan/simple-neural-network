import torch


class SimpleNeuralNetwork:
    def __init__(self):
        # Initialize weights for the hidden layer (2 neurons, 3 inputs)
        self.Wh = torch.tensor([[0.3, 0.3, 0.0],
                                [0.4, -0.5, 1.0]], dtype=torch.float32)  # Shape: (2, 3)
        # Initialize weights for the output layer (1 neuron, 2 inputs from hidden layer)
        self.Wout = torch.tensor([-1.0, 1.0], dtype=torch.float32)       # Shape: (2,)

    @staticmethod
    def activation(x):
        """
        Activation function that applies a threshold at 0.5.
        Returns 0 if x < 0.5, else returns 1.
        """
        return 0 if x < 0.5 else 1

    def forward(self, inputs):
        """
        Performs a forward pass through the network.
        :param inputs: A tensor of input features.
        :return: The output after passing through the network and activation function.
        """
        # Compute the input to the hidden layer neurons
        Zh = torch.mv(self.Wh, inputs)
        print(f"Sum values at hidden layer neurons: {Zh}")

        # Apply activation function to hidden layer outputs
        Uh = torch.tensor([self.activation(x) for x in Zh], dtype=torch.float32)
        print(f"Activated outputs from hidden layer neurons: {Uh}")

        # Compute the input to the output neuron
        Zout = torch.dot(self.Wout, Uh)
        print(f"Sum value at output neuron before activation: {Zout}")

        # Apply activation function to get the final output
        Y = self.activation(Zout)
        print(f"Final output after activation: {Y}")

        return Y


# Check if CUDA is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Example input features
house = 1    # Feature 1
rock = 0     # Feature 2
attr = 1     # Feature 3

# Convert input features to a tensor
input_features = torch.tensor([house, rock, attr], dtype=torch.float32)

# Instantiate the neural network
network = SimpleNeuralNetwork()

# Perform a forward pass with the input features
result = network.forward(input_features)

# Interpret the result
if result == 1:
    print("You are likely.")
else:
    print("Call me later.")
