"""
Neural Network module for Providing lazy access to following classes:
    Dense
    FCNN
    Trainer
    SquaredError
    Sigmoid
    Tanh
    Linear
"""

__all__ = [
    "Dense",
    "FCNN",
    "Trainer",
    "SquaredError",
    "Sigmoid",
    "Tanh",
    "Linear",
]


def __getattr__(name):

    if name == "Dense":
        from .layers import Dense
        return Dense

    if name == "FCNN":
        from .model import FCNN
        return FCNN

    if name == "Trainer":
        from .trainer import Trainer
        return Trainer

    if name == "SquaredError":
        from .losses import SquaredError
        return SquaredError

    if name == "Logistic":
        from .activations import Logistic
        return Logistic

    if name == "Tanh":
        from .activations import Tanh
        return Tanh

    if name == "Linear":
        from .activations import Linear
        return Linear

    raise AttributeError(
        f"module 'NN' has no attribute '{name}'"
    )
