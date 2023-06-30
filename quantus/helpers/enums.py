from enum import Enum


class DataType(Enum):
    """
    This enum represents the different types of data that a metric implementation currently supports.
    - IMAGE: Represents image data.
    - TABULAR: Represents tabular data.
    - TEXT: Represents text data.
    """
    IMAGE = "image" # 3D data
    TIMESERIES = "time-series" # 1D data
    TABULAR = "tabular" # 2D data
    TEXT = "text"


class ModelType(Enum):
    """
    This enum represents the different types of models that a metric can work with.
    - TORCH: Represents PyTorch models.
    - TF: Represents TensorFlow models.
    """
    TORCH = "torch"
    TF = "tensorflow"


class ScoreDirection(Enum):
    """
    This enum represents the direction that the score of a metric should go in for better results.
    - HIGHER: Higher scores are better.
    - LOWER: Lower scores are better.
    """
    HIGHER = "higher"
    LOWER = "lower"
