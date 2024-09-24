import torch
from torch import nn


class FlexibleModel(nn.Module):
    """
    A Flexible Feedforward Neural Network (FNN) with customizable layers and activation functions.

    Parameters:
    - number_of_layers: Number of layers in the network (default is 4).
    - activation_functions: A list of activation function names (default is ['ReLU', 'ReLU', 'ReLU', 'ReLU']).
    - input_shape: The number of input features (default is 124).
    - hidden_units: A list defining the number of units in each hidden layer (default is [10, 10, 10, 10]).
    """
    def __init__(self, number_of_layers: int = 4, activation_functions: list = ['ReLU', 'ReLU', 'ReLU', 'ReLU'], input_shape: int = 124, hidden_units: list = [10, 10, 10, 10]):
        super().__init__()

        # Check if length of activation_functions is the same as hidden_units
        self.check_len_activation_hidden_units(hidden_units, activation_functions)

        # Define layer list
        layers = []

        # Append the first layer
        layers.append(nn.Linear(in_features=input_shape, out_features=hidden_units[0]))
        layers.append(getattr(nn, activation_functions[0])())  # Instantiate first activation function

        # Loop and add all subsequent layers
        for i in range(1, number_of_layers):
            try:
                layers.append(nn.Linear(in_features=hidden_units[i-1], out_features=hidden_units[i]))
                layers.append(getattr(nn, activation_functions[i])())  # Instantiate activation function
            except AttributeError as e:
                raise ValueError(f"Invalid activation function: {activation_functions[i]}") from e
            except Exception as e:
                print(f"Something went wrong in layer activation: {e}")

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def check_len_activation_hidden_units(self, hidden_units, activation_functions):
        if len(hidden_units) != len(activation_functions):
            raise ValueError("The length of hidden_units must be the same as the length of activation_functions.")

if __name__ == "__main__":
    # Create Instance
    try:
        model = SameActivationFlexibleModel(number_of_layers=4, input_shape=124, hidden_units=[10, 20, 30, 10])
        print(model)  # Print the model structure to confirm it's created successfully
    except Exception as e:
        print(f"Something went wrong: {e}")
