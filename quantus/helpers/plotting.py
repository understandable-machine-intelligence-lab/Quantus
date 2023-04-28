"""This module provides some plotting functionality."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
from __future__ import annotations

from typing import List, Union, Dict, Any, Optional, Tuple, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from quantus.helpers import warn
from quantus.helpers.q_types import Explanation, FlipTask
from quantus.helpers.collection_utils import value_or_default


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
    results: Mapping[str, Mapping[str, Sequence[float]]],
    methods: Sequence[str] | None = None,
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


# ---------------- NLP --------------------


DEFAULT_SPECIAL_TOKENS = [
    "[CLS]",
    "[SEP]",
    "[PAD]",
]


class ColorMapper:
    """
    - Highest score get red (255,0,0).
    - Lowest score gets blue (0,0,255).
    - Positive scores are linearly interpolated between red and white (255, 255, 255).
    - Negative scores are linearly interpolated between blue and white (255, 255, 255).
    """

    def __init__(self, max_score: float, min_score: float):
        self.max_score = max_score
        self.min_score = min_score

    def to_rgb(
        self, score: float, normalize_to_1: bool = False
    ) -> Tuple[float, float, float]:
        k = 1.0 if normalize_to_1 else 255.0

        if score >= 0:
            red = k
            green = k * (1 - score / self.max_score)
            blue = k * (1 - score / self.max_score)
        else:
            red = k * (1 - abs(score / self.min_score))
            green = k * (1 - abs(score / self.min_score))
            blue = k
        return red, green, blue


def _create_div(
    explanation: Explanation,
    label: str,
    ignore_special_tokens: bool,
    special_tokens: List[str],
):
    # Create a container, which inherits root styles.
    div_template = """
        <div class="container">
            <p>
                {{label}} <br>
                {{saliency_map}}
            </p>
        </div>
        """

    # For each token, create a separate highlight span with different background color.
    token_span_template = """
        <span class="highlight-container" style="background:{{color}};">
            <span class="highlight"> {{token}} </span>
        </span>
        """
    tokens = explanation[0]
    scores = explanation[1]
    body = ""
    color_mapper = ColorMapper(np.max(scores), np.min(scores))

    for token, score in zip(tokens, scores):
        if ignore_special_tokens and token in special_tokens:
            continue
        red, green, blue = color_mapper.to_rgb(score)
        token_span = token_span_template.replace(
            "{{color}}", f"rgb({red},{green},{blue})"
        )
        token_span = token_span.replace("{{token}}", token)
        body += token_span + " "

    return div_template.replace("{{label}}", label).replace("{{saliency_map}}", body)


def visualise_explanations_as_html(
    explanations: List[Explanation],
    *,
    labels: Optional[List[str]] = None,
    ignore_special_tokens: bool = False,
    special_tokens: Optional[List[str]] = None,
) -> str:
    """
    Creates a heatmap visualisation from list of explanations. This method should be preferred for longer
    examples. It is rendered correctly in VSCode, PyCharm, Colab, however not in GitHub or JupyterLab.

    Parameters
    ----------
    explanations:
        List of tuples (tokens, salience) containing batch of explanations.
    labels:
        Optional, list of labels to display on top of each explanation.
    ignore_special_tokens:
        If True, special tokens will not be rendered in heatmap.
    special_tokens:
        List of special tokens to ignore during heatmap creation, default= ["[CLS]", "[END]", "[PAD]"].

    Returns
    -------

    html:
        string containing raw html to visualise explanations.

    Examples
    -------

    >>> from IPython.core.display import HTML
    >>> from quantus.helpers.plotting import visualise_explanations_as_html
    >>> a_batch = ...
    >>> raw_html = visualise_explanations_as_html(a_batch)
    >>> HTML(raw_html)

    """

    special_tokens = value_or_default(special_tokens, lambda: DEFAULT_SPECIAL_TOKENS)
    # Define top-level styles
    heatmap_template = """
        <style>

            .container {
                line-height: 1.4;
                text-align: center;
                margin: 10px 10px 10px 10px;
                color: black;
                background: white;
            }

            p {
                font-size: 16px;
            }

            .highlight-container, .highlight {
                position: relative;
                border-radius: 10% 10% 10% 10%;
            }

            .highlight-container {
                display: inline-block;
            }

            .highlight-container:before {
                content: " ";
                display: block;
                height: 90%;
                width: 100%;
                margin-left: -3px;
                margin-right: -3px;
                position: absolute;
                top: -1px;
                left: -1px;
                padding: 10px 3px 3px 10px;
            }

        </style>

        {{body}}
        """

    spans = ""
    # For each token, create a separate div holding whole input sequence on 1 line.
    for i, explanation in enumerate(explanations):
        label = labels[i] if labels is not None else ""
        div = _create_div(explanation, label, ignore_special_tokens, special_tokens)
        spans += div
    return heatmap_template.replace("{{body}}", spans)


def _value_at_index_or_default(values, index, default):
    if len(values) > index:
        return values[index]
    else:
        return default


def visualise_explanations_as_pyplot(
    explanations: List[Explanation],
    labels: Optional[List[str]] = None,
    v_len_scale=0.75,
    h_len_scale=1.25,
):
    """
    Plots attributions over a batch of text sequence explanations. This function should be preferred is you need your
    heatmaps to be correctly displayed in GitHubs preview. For longer inputs (over 15-20) tokens, the cells become
    smaller, and it could be hard for viewer to see the actual tokens.

    References:
        - https://stackoverflow.com/questions/74046734/plot-text-saliency-map-in-jupyter-notebook

    Parameters
    ----------
    explanations:
        List of tuples (tokens, salience) containing batch of explanations.
    labels:
        Optional labels to display above each row.

    Returns
    -------
    plot: matplotplib.pyplot.figure object, which will be automatically rendered by jupyter.
    """

    h_len = len(explanations)
    v_len = len(explanations[0][0])

    tokens = [i[0] for i in explanations]
    scores = [i[1] for i in explanations]

    fig, axes = plt.subplots(
        h_len,
        v_len,
        figsize=(v_len * v_len_scale, h_len * h_len_scale),
        gridspec_kw=dict(left=0.0, right=1.0),
    )
    hspace = 1.0 if labels is not None else 0.1
    plt.subplots_adjust(hspace=hspace, wspace=0.0)
    for i, ax in enumerate(axes):
        color_mapper = ColorMapper(np.max(scores[i]), np.min(scores[i]))
        if labels:
            ax[v_len // 2].set_title(labels[i])

        scores_row = scores[i]
        tokens_row = tokens[i]
        for j in range(v_len):
            score = _value_at_index_or_default(scores_row, j, 0.0)
            token = _value_at_index_or_default(tokens_row, j, "")
            color = color_mapper.to_rgb(score, normalize_to_1=True)
            rect = plt.Rectangle((0, 0), 1, 1, color=color)
            ax[j].add_patch(rect)
            ax[j].text(0.5, 0.5, token, ha="center", va="center")
            ax[j].set_xlim(0, 1)
            ax[j].set_ylim(0, 1)
            ax[j].axis("off")
            ax[j] = fig.add_axes([0, 0.05, 1, 0.9], fc=[0, 0, 0, 0])

    ax = axes.ravel()[-1]
    for axis in ["left", "right"]:
        ax.spines[axis].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def plot_token_flipping_experiment(
    score: np.ndarray | List[np.ndarray],
    task: FlipTask = "pruning",
    legend: List[str] | None = None,
    style: Dict[str, ...] | None = None,
) -> plt.Axes:
    if isinstance(score, np.ndarray):
        score = [score]

    style = value_or_default(style, lambda: {})
    fig, axes = plt.subplots()

    axes.set(xlabel="% tokens flipped.")
    axes.set(ylabel="$(y_o - y')^2$")
    axes.set_title(f"Token {task} experiment")

    with plt.style.context(style):
        for i in score:
            if i.ndim == 2:
                i = np.mean(i, axis=0)
            num_tokens = len(i)

            x = np.arange(0, num_tokens + 1)
            i = np.concatenate([[0.0], i])

            axes.plot(i, x, marker="o")

        if legend is not None:
            axes.legend(legend)

    return axes
