"""This module implements the base class for creating evaluation measures."""
from typing import Optional, Any, Union, List, Dict
from termcolor import colored
import numpy as np
import time
import warnings
from ..helpers.utils import attr_check


class Metric:
    """
    This class is the base class for creating evaluation measures.
    The measures outputs at minimum one numerical value per sample and explanation method.
    If more than one sample and/or more than one explanation method is evaluated, then the dimensions increases.
    """

    @attr_check
    def __init__(
        self,
        **kwargs: dict
    ):
        """ Initialize Measure. """
        self.name = "Base Metric"


    def __call__(
        self,
        model,
        inputs: np.array,
        targets: Union[np.array, int],
        attributions: Union[np.array, None],
    ):
        """Placeholder to compute measure for given data and attributions.
        Return float/Array per Sample."""

    @property
    def interpret_scores(self):
        """
        What the output mean:
        What a high versus low value indicates:
        Assumptions (to be concerned about):
        Further reading:
        """
        print(self.__doc__) #print(self.__call__.__doc__)

    def print_warning(self, text: str = "") -> None:
        time.sleep(2)
        warnings.warn(colored(text=text, color="red"))

    #@property
    #def warning_text(self, text: str):
    #    raise NotImplementedError(f"Warning text need to be set.")

    @property
    def list_hyperparameters(self) -> dict:
        attr_exclude = ["args", "kwargs", "all_results", "last_results", "img_size", "nr_channels", "text"]
        return {k: v for k, v in self.__dict__.items() if k not in attr_exclude}

    def set_params(self, key: str, value: Any) -> dict:
        self.kwargs[key] = value
        return self.kwargs