"""This module provides some plotting functionality."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from typing import List, Union, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from quantus.helpers import warn


def plot_pixel_flipping_experiment(
    y_batch: np.ndarray,
    scores: List[Any],
    single_class: Union[int, None] = None,
    *args,
    **kwargs,
) -> None:
    """
    Plot the pixel-flipping experiment as done in paper:

    References:
        1) Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier
        decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.

    Parameters
    ----------
    y_batch: np.ndarray
         The list of true labels.
    scores: list
        The list of evalution scores.
    single_class: integer, optional
        An integer to specify the label to plot.
    args: optional
        Arguments.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    None
    """

    fig = plt.figure(figsize=(8, 6))
    if single_class is None:
        for c in np.unique(y_batch):
            indices = np.where(y_batch == c)
            plt.plot(
                np.linspace(0, 1, len(scores[0])),
                np.mean(np.array(scores)[indices], axis=0),
                label=f"target: {str(c)} ({indices[0].size} samples)",
            )
    plt.xlabel("Fraction of pixels flipped")
    plt.ylabel("Mean Prediction")
    plt.gca().set_yticklabels(
        ["{:.0f}%".format(x * 100) for x in plt.gca().get_yticks()]
    )
    plt.gca().set_xticklabels(
        ["{:.0f}%".format(x * 100) for x in plt.gca().get_xticks()]
    )
    plt.legend()
    plt.show()


def plot_selectivity_experiment(results: Dict[str, List[Any]], *args, **kwargs) -> None:
    """
    Plot the selectivity experiment as done in paper:

    References:
        1) Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller.
        "Methods for interpreting and understanding deep neural networks."
        Digital Signal Processing 73 (2018): 1-15.

    Parameters
    ----------
    results: list, dict
        The results fromm the Selectivity experiment(s).
    args: optional
        Arguments.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    None
    """

    fig = plt.figure(figsize=(8, 6))
    if isinstance(results, dict):
        alllengths = [len(score) for scores in results.values() for score in scores]
        minlength = np.min(alllengths)
        if np.any(np.array(alllengths) > minlength):
            warn.warn_different_array_lengths()
        for method, scores in results.items():
            plt.plot(
                np.arange(0, len(scores[0][:minlength])),
                np.mean(np.array([score[:minlength] for score in scores]), axis=0),
                label=f"{str(method.capitalize())}",
            )
    elif isinstance(results, list):
        plt.plot(np.arange(0, len(results)), np.mean(results, axis=0))
    plt.xlabel(f"# Patches removed")
    plt.ylabel(f"Average function value $f(x)$")
    plt.gca().set_yticklabels(
        ["{:.0f}%".format(x * 100) for x in plt.gca().get_yticks()]
    )
    plt.legend()
    plt.show()


def plot_region_perturbation_experiment(
    results: Dict[str, List[Any]], *args, **kwargs
) -> None:
    """
    Plot the region perturbation experiment as done in paper:

    References:
        1) Samek, Wojciech, et al. "Evaluating the visualization of what a deep
        neural network has learned." IEEE transactions on neural networks and
        learning systems 28.11 (2016): 2660-2673.

    Parameters
    ----------
    results: list, dict
        The results fromm the Selectivity experiment(s).
    args: optional
        Arguments.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    None
    """

    fig = plt.figure(figsize=(8, 6))
    if isinstance(results, dict):
        alllengths = [len(score) for scores in results.values() for score in scores]
        minlength = np.min(alllengths)
        if np.any(np.array(alllengths) > minlength):
            warn.warn_different_array_lengths()
        for method, scores in results.items():
            plt.plot(
                np.arange(0, len(scores[0][:minlength])),
                np.mean(np.array([score[:minlength] for score in scores]), axis=0),
                label=f"{str(method.capitalize())}",
            )
    else:
        plt.plot(np.arange(0, len(results)), np.mean(results, axis=0))
    plt.xlabel("Perturbation steps")
    plt.ylabel("AOPC relative to random")
    plt.gca().set_yticklabels(
        ["{:.0f}%".format(x * 100) for x in plt.gca().get_yticks()]
    )
    plt.legend()
    plt.show()


def plot_sensitivity_n_experiment(
    results: Union[List[float], Dict[str, List[float]]], *args, **kwargs
) -> None:
    """
    Plot the sensitivity n experiment as done in paper:

    References:
        1) Ancona, Marco, et al. "Towards better understanding of gradient-based attribution
        methods for deep neural networks." arXiv preprint arXiv:1711.06104 (2017).

    Parameters
    ----------
    results: list, dict
        The results fromm the Selectivity experiment(s).
    args: optional
        Arguments.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    None
    """

    fig = plt.figure(figsize=(8, 6))
    if isinstance(results, dict):
        for method, scores in results.items():
            plt.plot(
                np.arange(0, len(scores)),
                scores,
                label=f"{str(method.capitalize())}",
            )
    else:
        plt.plot(np.linspace(0, 1, len(results)), results)
    plt.xlabel(f"$n$")
    plt.ylabel(f"Correlation coefficient")
    plt.gca().set_yticklabels(
        ["{:.0f}%".format(x * 100) for x in plt.gca().get_yticks()]
    )
    plt.legend()
    plt.show()


def plot_model_parameter_randomisation_experiment(
    results: Dict[str, dict],
    methods=None,
    *args,
    **kwargs,
) -> None:
    """
    Plot the model parameter randomisation experiment as done in paper:

    References:
        1) Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., and Kim, B. "Sanity Checks for Saliency Maps."
        arXiv preprint, arXiv:1810.073292v3 (2018)

    Parameters
    ----------
    results: list, dict
        The results fromm the Selectivity experiment(s).
    args: optional
        Arguments.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    None
    """

    fig = plt.figure(figsize=(8, 6))

    if methods:
        for method in methods:
            for _ in results[method]:
                layers = list(results[method].keys())
                scores: Dict[Any, Any] = {k: [] for k in layers}
                # samples = len(results[method])
                # for s in range(samples):
                for layer in layers:
                    scores[layer].append(results[method][layer])

            plt.plot(layers, [np.mean(v) for k, v in scores.items()], label=method)
    else:

        layers = list(results.keys())
        scores = {k: [] for k in layers}
        # samples = len(results)
        # for s in range(samples):
        for layer in layers:
            scores[layer].append(results[layer])

        plt.plot(layers, [np.mean(v) for k, v in scores.items()])

    plt.xticks(rotation=90)
    plt.xlabel("Layers")
    plt.ylabel(kwargs.get("similarity_metric", "Score"))

    if methods:
        plt.legend(methods)
    plt.show()


def plot_focus(
    results: Dict[str, List[float]],
    *args,
    **kwargs,
) -> None:
    """
    Plot the Focus experiment as done in the paper:

    References:
        1) Arias-Duart, Anna, et al. 'Focus! Rating XAI Methods
        and Finding Biases. arXiv:2109.15035 (2022)"

    Parameters
    ----------
    results: dict
        A dictionary with the Focus scores obtained using different feature attribution methods.
    args: optional
        Arguments.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    None
    """

    fig = plt.figure(figsize=(8, 6))
    for method, scores in results.items():
        plt.boxplot(scores)
        plt.xlabel(method)

    plt.ylabel("Focus score")
    plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    plt.show()
