"""This module implements the base class for creating evaluation measures."""
from typing import Optional, Any, Union, List, Dict, Callable
from termcolor import colored
import numpy as np
import time
import warnings
from ..helpers.utils import attr_check
import matplotlib.pyplot as plt

class Metric:
    """
    Implementation of [NAME] metric by [AUTHORS NAME AND YEAR].

    [ONE/ TWO SENTENCES DESCRIPTION OF TEST].
    For a [FULL or MATHEMATICAL (if included in the paper)] definition of the [NAME] metric, see References.

    The scores range between [RANGES], where higher scores are typically (but not necessarily) [BETTER/WORSE].

    The [NAME] metric intend to capture an explanation's relative [CATEGORY]. Other metrics that
    belong to this category are [METRIC_CATEGORY_1], [METRIC_CATEGORY_2] and [METRIC_CATEGORY_N].

    References.

        [CITATION]

    Further notes.

        The [NAME] metric assumes that [METHOD ASSUMPTION 1] is true. Also, [NAME] metric assumes that [METHOD
        ASSUMPTION 1].

        Further, the [NAME] metric is known to be sensitive to the choice of [HYPERPARAMTER_1], [HYPERPARAMTER_2] and
        [HYPERPARAMTER_N]. To avoid any misinterpretation of the results, go over and select each hyperparameter of the
        metric carefully. Query .list_hyperparameters of the metric instance to view all the hyperparameters.

        The [NAME] metric was originally introduced in the context of [APPLICATION/DOMAIN/DATASET]. Pay attention to
        what extent your test domain differ from this and whether the [NAME] metric makes sense in your application.

    Changes amd/ or additions.

        In addition to the original implementation, for [REASON FOR ADDITION], we have implemented [THIS] and [THAT].

        Since information about the [THIS HYPERPARAMETER] was missing from the original publication, we have assumed that
        the default choice of the [THIS HYPERPARAMETER] is [VALUE] and set it accordingly. Other choices for the
        [THIS HYPERPARAMETER] is possible and can be changed to any option as listed in [
        AVAILABLE_PERTURBATION_FUNCTIONS/ AVAILABLE_SIMILARITY_FUNCTIONS] or as user-defined.

    """

    @attr_check
    def __init__(self, *args, **kwargs):
        """ Initialize Measure. """
        self.name = "Base Metric"

        self.text_warning = f"""\n\nThe [METRIC NAME] metric is known to be sensitive to the choice of baseline value 
        'perturb_baseline', size of subset |S| 'subset_size' and the number of runs (for each input and explanation 
        pair) 'nr_runs'. \nGo over and select each hyperparameter of the [METRIC NAME] metric carefully to avoid 
        misinterpretation of scores. \nTo view all relevant hyperparameters call .list_hyperparameters method. \nFor 
        more reading, please see [INSERT CITATION]."""

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

    
    def plot(self,
             plot_func: Callable,
             show: bool = True,
             path_to_save: Union[str, None] = None,
             *args,
             **kwargs) -> None:

        assert callable(plot_func), "Make sure that 'plot_func' is a callable."

        plot_func(*args, **kwargs)

        if show:
            plt.show()

        if path_to_save:
            plt.savefig(fname=path_to_save, dpi=400)

        return None

if __name__ == '__main__':

    # Run tests!
    pass