"""This module implements the base class for creating evaluation measures."""
import warnings
import numpy as np
from typing import Union, List, Dict, Any
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
        self.normalise = self.kwargs.get("normalise", False)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_max)
        self.default_plot_func = Callable
        self.text_warning = f"""\n\nThe [METRIC NAME] metric is likely to be sensitive to the choice of baseline value 
        'perturb_baseline', size of subset |S| 'subset_size' and the number of runs (for each input and explanation 
        pair) 'nr_runs'. \nGo over and select each hyperparameter of the [METRIC NAME] metric carefully to avoid 
        misinterpretation of scores. \nTo view all relevant hyperparameters call .get_params method. \nFor 
        more reading, please see [INSERT CITATION]."""
        self.last_results = []
        self.all_results = []

        # Print warning at metric initialisation.
        #self.print_warning(text=self.text_warning)


    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: Union[np.ndarray, int],
        a_batch: Union[np.ndarray, None],
        s_batch: Union[np.ndarray, None],
        *args,
        **kwargs,
    ) -> Union[int, float, list, dict]:
        """

        Parameters
        ----------
        model: a torch model that is to-be-explained.
        x_batch: a np.ndarray which contains the inputs that are to-be-explained.
        y_batch: a Union[np.ndarray, int] which contains the outputs that are to-be-explained.
        a_batch: a Union[np.ndarray, None] which contains pre-computed attributions.
        s_batch: a Union[np.ndarray, None] which contains segmentation masks that matches the input.
        args: optional arguments.
        kwargs: optional dictionary with arguments (key, value) pairs.

        Returns: Union[int, float, list, dict] the output of the metric, which may differ depending on implemntation.
        -------

        """
        pass

    @property
    def interpret_scores(self) -> None:
        """

        Returns
        -------

        """

        print(self.__call__.__doc__.split("callable.")[1].split("Parameters")[0])

    @property
    def get_params(self) -> dict:
        """
        List parameters of metric.
        Returns: a dictionary with attributes if not excluded from pre-determined list.
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
        Set a parameter of a metric.
        Parameters
        ----------
        key: attribute of metric to mutate.
        value: value to update the key with

        -------
        Returns: the updated dictionary.

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
        Basic plotting functionality for Metric class.
        The user provides a plot_func (Callable) that contains the actual plotting logic (but returns None).

        Parameters
        ----------
        plot_func: a Callable with the actual plotting logic.
        show: a boolean to state if the plot shall be shown.
        path_to_save: a string that specifies the path to save file.
        args: an optional with additional arguments.
        kwargs: an optional dict with additional arguments.

        Returns: None.
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
