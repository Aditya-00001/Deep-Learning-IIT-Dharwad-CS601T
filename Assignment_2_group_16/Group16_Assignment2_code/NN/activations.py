import numpy as np

class Logistic:

    def __init__(self):
        self.activation_value = None

    def forward(self, activation_value):
        self.activation_value = 1 / (1 + np.exp(-activation_value))
        return self.activation_value

    def backward(self, dA):
        return dA * self.activation_value * (1 - self.activation_value)

class Tanh:

    def __init__(self):
        self.activation_value = None

    def forward(self, activation_value):
        self.activation_value = np.tanh(activation_value)
        return self.activation_value

    def backward(self, dA):
        return dA * (1 - self.activation_value**2)

class Linear:
    def __init__(self):
        self.activation_value = None
        
    def forward(self,activation_value):
        self.activation_value = activation_value
        return self.activation_value
    
    def backward(self, dA):
        return dA