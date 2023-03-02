import unittest
from ldimbenchmark.classes import Hyperparameter


class Validation(unittest.TestCase):
    def test_normal_usage(self):
        hyperparameter_int_minmax = Hyperparameter(
            name="test",
            description="test",
            value_type=int,
            default=1,
            min=0,
            max=10,
        )

        hyperparameter_int_options = Hyperparameter(
            name="test",
            description="test",
            value_type=int,
            default=1,
            options=[1, 2, 3],
        )

        hyperparameter_float_minmax = Hyperparameter(
            name="test",
            description="test",
            value_type=float,
            default=1.0,
            min=0.0,
            max=10.0,
        )

        hyperparameter_float_options = Hyperparameter(
            name="test",
            description="test",
            value_type=float,
            default=1.0,
            options=[1.0, 2.0, 3.0],
        )

        hyperparameter_bool = Hyperparameter(
            name="test",
            description="test",
            value_type=bool,
            default=True,
        )

        hyperparameter_string = Hyperparameter(
            name="test",
            description="test",
            value_type=str,
            default="option1",
            options=["option1", "option2"],
        )

    def test_error_usage(self):
        # Hyperparameter with value_type int or float must not have options and min and max
        with self.assertRaises(Exception) as context:
            hyperparameter = Hyperparameter(
                name="test",
                description="test",
                value_type=int,
                options=[1, 2, 3],
                min=0,
                max=10,
            )

            hyperparameter = Hyperparameter(
                name="test",
                description="test",
                value_type=float,
                options=[1, 2, 3],
                min=0,
                max=10,
            )

        # Hyperparameter with value_type bool must not have options and min and max
        with self.assertRaises(Exception) as context:
            hyperparameter = Hyperparameter(
                name="test",
                description="test",
                value_type=bool,
                options=[1, 2, 3],
                min=0,
                max=10,
            )

        # Hyperparameter default value does not match value_type
        with self.assertRaises(Exception) as context:
            hyperparameter = Hyperparameter(
                name="test",
                description="test",
                value_type=int,
                default="1",
            )
