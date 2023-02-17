import unittest
from ldimbenchmark.classes import Hyperparameter


class MyTestCase(unittest.TestCase):
    def test_cusum_normal_usage(self):
        hyperparameter_int_minmax = Hyperparameter(
            name="test",
            description="test",
            type=int,
            default=1,
            min=0,
            max=10,
        )

        hyperparameter_int_options = Hyperparameter(
            name="test",
            description="test",
            type=int,
            default=1,
            options=[1, 2, 3],
        )

        hyperparameter_float_minmax = Hyperparameter(
            name="test",
            description="test",
            type=float,
            default=1,
            min=0,
            max=10,
        )

        hyperparameter_float_options = Hyperparameter(
            name="test",
            description="test",
            type=float,
            default=1,
            options=[1, 2, 3],
        )

        hyperparameter_bool = Hyperparameter(
            name="test",
            description="test",
            type=bool,
            default=True,
        )

        hyperparameter_string = Hyperparameter(
            name="test",
            description="test",
            type=str,
            default="option1",
            options=["option1", "option2"],
        )

    def test_cusum_error_usage(self):
        # Hyperparameter with type str must have options
        with self.assertRaises(Exception) as context:
            hyperparameter = Hyperparameter(
                name="test",
                description="test",
                type=str,
            )

        # Hyperparameter with type int or float must have min and max or options
        with self.assertRaises(Exception) as context:
            hyperparameter = Hyperparameter(
                name="test",
                description="test",
                type=int,
            )

            hyperparameter = Hyperparameter(
                name="test",
                description="test",
                type=float,
            )

        # Hyperparameter with type int or float must not have options and min and max
        with self.assertRaises(Exception) as context:
            hyperparameter = Hyperparameter(
                name="test",
                description="test",
                type=int,
                options=[1, 2, 3],
                min=0,
                max=10,
            )

            hyperparameter = Hyperparameter(
                name="test",
                description="test",
                type=float,
                options=[1, 2, 3],
                min=0,
                max=10,
            )

        # Hyperparameter with type bool must not have options and min and max
        with self.assertRaises(Exception) as context:
            hyperparameter = Hyperparameter(
                name="test",
                description="test",
                type=bool,
                min=0,
                max=10,
            )

            hyperparameter = Hyperparameter(
                name="test",
                description="test",
                type=bool,
                options=[1, 2, 3],
            )

        # Hyperparameter default value does not match type
        with self.assertRaises(Exception) as context:
            hyperparameter = Hyperparameter(
                name="test",
                description="test",
                type=int,
                default="1",
            )

        # self.assertTrue("This is broken" in str(context.exception))
