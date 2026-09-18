import numpy as np

class Dense:

    def __init__(
        self,
        input_dimension,
        output_dimension,
        weight_range=None
    ):

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        # Weight initialization
        if weight_range is None:

            scale = 1 / np.sqrt(input_dimension)

            self.weights = np.random.randn(
                1 + input_dimension,
                output_dimension
            ) * scale

        else:

            self.weights = np.random.uniform(
                -weight_range,
                weight_range,
                (1 + input_dimension, output_dimension)
            )

        self.X = None
        self.Z = None
        self.dW = None

    def forward(self, x):

        self.X = np.concatenate(([1], x))

        self.Z = np.dot(self.X, self.weights)

        return self.Z

    def backward(self, dZ):

        self.dW = np.outer(self.X, dZ)

        dX = dZ @ self.weights.T

        return dX[1:]