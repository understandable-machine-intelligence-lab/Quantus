"""This module implements the base class for creating evaluation measures."""
from typing import Optional, Any, Union, List
import numpy as np

class Measure:
    """
    This class is the base class for creating evaluation measures.
    The measures outputs at minimum one numerical value per sample and explanation method.
    If more than one sample and/or more than one explanation method is evaluated, then the dimensions increases.
    """

    def __init__(self,
                 #name: Optional[str] = "Measure",
                 **kwargs: dict):
        """ Initialize Measure. """
        #assert isinstance(name, str)
        assert isinstance(kwargs, dict)
        #self.name = name
        self.kwargs = kwargs

    def __call__(self,
                 model,
                 inputs: np.array,
                 targets: Union[np.array, int],
                 attributions: Union[np.array, None],
                 ):
        """Placeholder to compute measure for given data and attributions. Return float/Array per Sample. """

        raise NotImplementedError("Implementation of the Measure missing.")

    #def __str__(self):
    #    return '{self.name}'.format(self=self)
        #raise NotImplementedError("Name of the of the Measure missing.")

    @property
    def get_params(self) -> Dict[str, Union[Optional[str], Optional[int], Optional[float], List]]:
        for k, v in self.params.items():
            print(k, v)
        return self.params

    def set_params(self, key: Optional[str], value: Any) -> Dict[str, Any]:
        self.params[key] = value
        return self.params


class SequentialMeasure:

    def __init__(self,
                 *args: [Measure]):

        for measure in args:
            assert isinstance(measure, Measure)
        self.measures = [measure for measure in args]

    def __call__(self,
                 model,
                 inputs: np.array,
                 targets: Union[np.array, int],
                 attributions: Union[np.array, None]
                 ):

        results = {}

        for measure in self.measures:

            results[measure] = measure(model, inputs, targets, attributions)

        return results

    @property
    def get_measures(self) -> List[Measure]:

        return self.measures
