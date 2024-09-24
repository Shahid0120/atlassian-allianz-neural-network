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

    allowed_activation = [
        'ELU', 'Hardshrink', 'Hardsigmoid', 'Hardtanh', 'Hardswish',
        'LeakyReLU', 'LogSigmoid', 'MultiheadAttention', 'PReLU', 'ReLU',
        'ReLU6', 'RReLU', 'SELU', 'CELU', 'GELU', 'Sigmoid', 'SiLU',
        'Mish', 'Softplus', 'Softshrink', 'Softsign', 'Tanh',
        'Tanhshrink', 'Threshold', 'GLU'
    ]

    def __init__(self, number_of_layers: int = 4, activation_functions: list = None,
                 input_shape: int = 124, hidden_units: list = [10, 10, 10, 10]):
        super().__init__()
        
        if activation_functions is None:
            activation_functions = ['ReLU'] * number_of_layers
        
        self.check_len_activation_hidden_units(hidden_units, activation_functions)
        self.check_proper_activation_functions(activation_functions)

        # Define layer list
        layers = []

        # Append the first layer
        layers.append(nn.Linear(in_features=input_shape, out_features=hidden_units[0]))
        layers.append(getattr(nn, activation_functions[0])())

        # Loop and add all subsequent layers
        for i in range(1, number_of_layers):
            layers.append(nn.Linear(in_features=hidden_units[i - 1], out_features=hidden_units[i]))
            layers.append(getattr(nn, activation_functions[i])())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @staticmethod
    def check_len_activation_hidden_units(hidden_units, activation_functions) -> None:
        if len(hidden_units) != len(activation_functions):
            raise ValueError("The length of hidden_units must be the same as the length of activation_functions.")
        
    @staticmethod
    def check_proper_activation_functions(activation_functions) -> None:
        for func in activation_functions:
            if func not in FlexibleModel.allowed_activation:
                raise ValueError(f"{func} is not a valid Activation")

if __name__ == "__main__":
    # Create Instance
    try:
        model = FlexibleModel(number_of_layers=4, input_shape=124, hidden_units=[10, 20, 30, 10])
        print(model) 
    except Exception as e:
        print(f"Something went wrong: {e}")


