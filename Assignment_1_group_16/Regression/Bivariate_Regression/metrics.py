import numpy as np
class Metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    
    def mean_squared_error(self):
        return np.mean((self.y_true - self.y_pred) ** 2)

    def root_mean_squared_error(self):
        return np.sqrt(self.mean_squared_error())
    
    def percentage_root_mean_squared_error(self):
        return 100 * self.root_mean_squared_error() / np.mean(self.y_true)