"""This module implements the base class for creating evaluation measures."""
from typing import Optional, Any, Union, List, Dict
import numpy as np
from ..helpers.utils import attr_check


class Measure:
    """
    This class is the base class for creating evaluation measures.
    The measures outputs at minimum one numerical value per sample and explanation method.
    If more than one sample and/or more than one explanation method is evaluated, then the dimensions increases.
    """

    @attr_check
    def __init__(
        self,
        # name: Optional[str] = "Measure",
        **kwargs: dict
    ):
        """ Initialize Measure. """
        # assert isinstance(name, str)
        assert isinstance(kwargs, dict)
        self.name = "Base"
        self.kwargs = kwargs

    def __call__(
        self,
        model,
        inputs: np.array,
        targets: Union[np.array, int],
        attributions: Union[np.array, None],
    ):
        """Placeholder to compute measure for given data and attributions.
        Return float/Array per Sample."""

        raise NotImplementedError("Implementation of the Measure is missing.")

    @property
    def HOWTOREADSCORES(self):
        """
        What the output mean:
        What a high versus low value indicates:
        Assumptions (to be concerned about):
        Further reading:
        """
        print(self.__doc__)
        #print(self.__call__.__doc__)

    def __str__(self):
        pass
        # return '{self.name}'.format(self=self)
        # NotImplementedError("Name of the Measure is missing."):

    @property
    def get_params(
        self,
    ) -> Dict[str, Union[Optional[str], Optional[int], Optional[float], List]]:
        return self.__dict__
        # for k, v in self.kwargs.items():
        #    print(k, v)

    #

    def set_params(self, key: Optional[str], value: Any) -> Dict[str, Any]:
        self.kwargs[key] = value
        return self.kwargs
