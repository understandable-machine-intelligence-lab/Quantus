# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import abc
import warnings

from quantus.metrics.base import Metric
from quantus.metrics.base_perturbed import PerturbationMetric


"""Aliases to smoothen transition to uniform metric API."""


class BatchedMetric(Metric, abc.ABC):

    """Alias to quantus.Metric, will be removed in next major release."""

    def __subclasscheck__(self, subclass):
        warnings.warn(
            "BatchedMetric was deprecated, since it is just an alias to Metric. Please subclass Metric directly."
        )


class BatchedPerturbationMetric(PerturbationMetric, abc.ABC):
    """Alias to quantus.PerturbationMetric, will be removed in next major release."""

    def __subclasscheck__(self, subclass):
        warnings.warn(
            "BatchedPerturbationMetric was deprecated, "
            "since it is just an alias to Metric. Please subclass PerturbationMetric directly."
        )
