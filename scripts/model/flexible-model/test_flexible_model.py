import unittest 
import torch 
from flexible_model_building_class_basic import FlexibleModel


class TestFlexibleClass(unittest.TestCase):

    def test_model_creation(self):
        model = FlexibleModel(number_of_layers=4, input_shape=124, hidden_units=[10, 20, 30, 10], activation_functions=['ReLU', "ReLU", "ReLU", "ReLU"])
        self.assertIsInstance(model, FlexibleModel)

    def test_activation_requirements(self):
        with self.assertRaises(ValueError):
            model = FlexibleModel(number_of_layers=4, input_shape=124, hidden_units=[10, 20, 30, 10], activation_functions=['ReLU', "PeLu", "ReLu", "ReLU"]) # PeLu is not activation


    def test_len_hidden_units_activation(self):
        with self.assertRaises(ValueError):
            model = FlexibleModel(number_of_layers=4, input_shape=124, hidden_units=[10, 20, 30, 10, 10], activation_functions=['ReLU', "ReLU", "ReLU", "ReLU"]) # hidden has 1 more index

    def test_forward_pass(self):
        model = FlexibleModel(number_of_layers=4, input_shape=124, hidden_units=[10, 20, 30, 10], activation_functions=['ReLU', "ReLU", "ReLU", "ReLU"])
        dummy_input = torch.randn(5, 124)  # Batch size of 5
        output = model(dummy_input)
        self.assertEqual(output.shape, (5, 10))  # Check output shape is correct

if __name__ == "__main__":
    unittest.main()
