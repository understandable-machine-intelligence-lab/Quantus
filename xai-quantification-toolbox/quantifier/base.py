"""This module implements the quantifier which evaluation measures."""
from typing import Union, List
import tensorflow as tf
import torch
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