"""This module implements the base class for creating evaluation measures."""
import time
import warnings
import numpy as np
from typing import Union, List, Dict, Any
from termcolor import colored
import matplotlib.pyplot as plt
from ..helpers.utils import *
from ..helpers.asserts import *
from ..helpers.plotting import *
from ..helpers.normalise_func import *
from ..helpers.warn_func import *


class Metric:
    """
    Implementation base Metric class.
    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """Initialize Metric."""
        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_max)
        self.default_plot_func = Callable
        self.text_warning = f"""\n\nThe [METRIC NAME] metric is known to be sensitive to the choice of baseline value 
        'perturb_baseline', size of subset |S| 'subset_size' and the number of runs (for each input and explanation 
        pair) 'nr_runs'. \nGo over and select each hyperparameter of the [METRIC NAME] metric carefully to avoid 
        misinterpretation of scores. \nTo view all relevant hyperparameters call .list_hyperparameters method. \nFor 
        more reading, please see [INSERT CITATION]."""
        self.last_results = []
        self.all_results = []

    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: Union[np.ndarray, int],
        a_batch: Union[np.ndarray, None],
        **kwargs,
    ) -> Union[int, float, list, dict]:
        """
        • What the metrics tests
        • What the output mean
        • What a high versus low value indicates


        Parameters
        ----------
        model (torch.nn, torchvision.models)
        x_batch (np.ndarray)
        y_batch (np.ndarray)
        a_batch (np.ndarray)
        kwargs (dict)

        Returns
        -------

        """

    @property
    def interpret_scores(self) -> None:
        """

        Returns
        -------

        """

        print(self.__call__.__doc__.split("callable.")[1].split("Parameters")[0])

    def print_warning(self, text: str = "") -> None:
        """

        Parameters
        ----------
        text

        Returns
        -------

        """
        time.sleep(2)
        warnings.warn(colored(text=text, color="blue"), category=Warning)

    @property
    def list_hyperparameters(self) -> dict:
        """

        Returns
        -------

        """
        attr_exclude = [
            "args",
            "kwargs",
            "all_results",
            "last_results",
            "img_size",
            "nr_channels",
            "text",
        ]
        return {k: v for k, v in self.__dict__.items() if k not in attr_exclude}

    def set_params(self, key: str, value: Any) -> dict:
        """

        Parameters
        ----------
        key
        value

        Returns
        -------

        """
        self.kwargs[key] = value
        return self.kwargs

    def plot(
        self,
        plot_func: Union[Callable, None] = None,
        show: bool = True,
        path_to_save: Union[str, None] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Plotting functionality for Metric class. The user provides a plot_func (Callable) that contains the
        actual plotting logic (but returns None).

        Parameters
        ----------
        plot_func
        show
        path_to_save
        args
        kwargs

        Returns
        -------

        """

        # Get plotting func if not provided.
        if plot_func is None:
            plot_func = kwargs.get("plot_func", self.default_plot_func)

        # Asserts.
        assert_plot_func(plot_func=plot_func)

        # Plot!
        plot_func(*args, **kwargs)

        if show:
            plt.show()

        if path_to_save:
            plt.savefig(fname=path_to_save, dpi=400)

        return None


if __name__ == "__main__":

    # Run tests!
    pass
