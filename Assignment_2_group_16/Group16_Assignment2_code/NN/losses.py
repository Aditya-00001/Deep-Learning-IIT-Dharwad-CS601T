import numpy as np

class SquaredError:
    def __init__(self):
        self.y_true = None
        self.y_pred = None
    
    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

        return 0.5 * np.sum((y_true - y_pred) ** 2)

    def backward(self):
        return self.y_pred - self.y_true