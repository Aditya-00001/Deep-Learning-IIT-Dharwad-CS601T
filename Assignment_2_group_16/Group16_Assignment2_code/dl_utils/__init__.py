__all__ = [
    "Data_preparation",
    "Plots",
    "Metrics"
]


def __getattr__(name):

    if name == "Data_preparation":
        from .data_utils import Data_preparation
        return Data_preparation

    elif name == "Plots":
        from .visualization_utils import Plots
        return Plots

    elif name == "Metrics":
        from .metrics import Metrics
        return Metrics

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
