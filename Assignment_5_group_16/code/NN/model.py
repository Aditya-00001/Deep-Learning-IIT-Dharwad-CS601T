import numpy as np
class FCNN:
    def __init__(self,layers):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad
    
    def predict(self, X):
        predictions = []

        for x in X:
            predictions.append(
                self.forward(x)
            )

        return np.array(predictions)
    
    def forward_with_activations(self, x):

        activations = []

        for layer in self.layers:

            x = layer.forward(x)

            if layer.__class__.__name__ in [
                "Logistic",
                "Tanh",
                "Linear"
            ]:
                activations.append(
                    x.copy()
                )

        return activations