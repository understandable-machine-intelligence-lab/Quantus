"""This module implements the quantifier which evaluation measures."""
from typing import Union, List
import tensorflow as tf
import torch
import numpy as np

from ..measures.base import Measure


class Quantifier:
    """
    This class is where evaluation specified and where it is run against some data, explanations and model.
    Here we specify how things are run (local, #gpus) and where the output is stored.
    """

    def __init__(self,
                 measures: List[Measure]):
        self.measures = measures


    def fit(self, inputs, targets, explanations=None):
        if explanations is None:
            get_explanations(inputs, tragets)
        for i in inputs:
            for e in explanations:
                for m in self.measures:
                    self. = m.fit(i, e)

    @staticmethod
    def get_explanations(self):
        pass


class ModularQuantifier:
    """
        This class is where evaluation specified and where it is run against some data, explanations and model.
        Here we specify how things are run (local, #gpus) and where the output is stored.
        """

    def __init__(self,
                 measure: Measure):
        self.measure = measure

    def fit(self, inputs, targets, model, attributions=None, save_to_file=False):
        if attributions is None:
            get_explanations(inputs, targets)

        if isinstance(inputs, np.ndarray):
            results = self.measure(model, inputs, targets, attributions)
            self.__save_results(self.measure, results, save_to_file=save_to_file)

        else:
            try:
                for batch in inputs:
                    # extract data and handles(fname) from dataloader
                    data, handles = extract_data(batch)
                    # load attributions
                    attributions = get_attributions(data, targets, handles)
                    # compute measures
                    results = self.measure(model, data, targets, attributions)
                    self.__save_results(self.measure, results, save_to_file=save_to_file)

            except TypeError:
                print("{} is not iterable.".format(inputs))

    def __save_results(self, measure, results, save_to_file=False):
        """ Save Measure results to internal representation. """
        raise NotImplementedError()

    @staticmethod
    def get_explanations(self):
        pass