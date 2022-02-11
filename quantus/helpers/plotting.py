"""This module provides some plotting functionality."""
from typing import List, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import *


def plot_pixel_flipping_experiment(
    y_batch: np.ndarray,
    scores: List[float],
    single_class: Union[int, None] = None,
    *args,
    **kwargs,
) -> None:
    """
    Plot the pixel-flippng experiment as done in paper:

    References:
        1) Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier
        decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.

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


def plot_selectivity_experiment(
    results: Union[List[float], Dict[str, List[float]]], *args, **kwargs
) -> None:
    """
    Plot the selectivity experiment as done in paper:

    References:
        1) Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller.
        "Methods for interpreting and understanding deep neural networks."
        Digital Signal Processing 73 (2018): 1-15.
    """
    fig = plt.figure(figsize=(8, 6))
    if isinstance(results, dict):
        for method, scores in results.items():
            plt.plot(
                np.arange(0, len(scores[0])),
                np.mean(np.array(list(scores.values())), axis=0),
                label=f"{str(method.capitalize())} ({len(list(scores))} samples)",
            )
    elif isinstance(results, list):
        pass
    plt.xlabel(f"# Patches removed")
    plt.ylabel(f"Average function value $f(x)$")
    plt.gca().set_yticklabels(
        ["{:.0f}%".format(x * 100) for x in plt.gca().get_yticks()]
    )
    plt.legend()
    plt.show()


def plot_region_perturbation_experiment(
    results: Union[List[float], Dict[str, List[float]]], *args, **kwargs
) -> None:
    """
    Plot the region perturbation experiment as done in paper:

     References:
        1) Samek, Wojciech, et al. "Evaluating the visualization of what a deep
         neural network has learned." IEEE transactions on neural networks and
          learning systems 28.11 (2016): 2660-2673.
    """
    fig = plt.figure(figsize=(8, 6))
    if isinstance(results, dict):
        for method, scores in results.items():
            plt.plot(
                np.arange(0, len(scores[0])),
                np.mean(np.array(list(scores.values())), axis=0),
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

    """
    fig = plt.figure(figsize=(8, 6))
    if isinstance(results, dict):
        for method, scores in results.items():
            plt.plot(
                np.linspace(0, 1, len(scores[0])),
                np.mean(np.array(list(scores.values())), axis=0),
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


def plot_superpixel_segments(
    img: np.ndarray, segments: np.ndarray, *args, **kwargs
) -> None:
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(
        mark_boundaries(
            np.reshape(img, (kwargs.get("img_size", 224), kwargs.get("img_size", 224))),
            segments,
            mode="subpixel",
        )
    )
    plt.title("Segmentation outcome")
    plt.grid(False)
    plt.show()


def plot_model_parameter_randomisation_experiment(
    results: Union[List[float], Dict[str, List[float]]],
    methods=None,
    *args,
    **kwargs,
) -> None:
    """
    Plot the model parameter randomization experiment as done in paper:
     References:
        1) Samek, Wojciech, et al. "Evaluating the visualization of what a deep
         neural network has learned." IEEE transactions on neural networks and
          learning systems 28.11 (2016): 2660-2673.
    """

    fig = plt.figure(figsize=(8, 6))

    if methods:
        for method in methods:
            for _ in results[method]:
                layers = list(results[method].keys())
                scores = {k: [] for k in layers}
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
