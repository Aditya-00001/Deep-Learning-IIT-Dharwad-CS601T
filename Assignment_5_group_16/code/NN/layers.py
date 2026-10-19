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

class Convolution2D:
    
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        scale = np.sqrt(2/ (input_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * scale
        self.biases = np.zeros((output_channels, 1))
        pass
    def convolution():
        
        pass
    
class MaxPooling2D:
    
    def __init__(self, pool_size, stride=None):
        pass