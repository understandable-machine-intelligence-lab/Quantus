from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from quantus.functions.loss_func import mse
from quantus.nlp.helpers.types import Explanation
from quantus.nlp.helpers.utils import value_or_default

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
    Creates a heatmap visualisation from list of explanations.

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
    >>> from quantus.nlp import visualise_explanations_as_html
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


def visualise_explanations_as_pyplot(
    explanations: List[Explanation], labels: Optional[List[str]] = None
):
    """
    Plots attributions over a batch of text sequence explanations.

    References:
        - https://stackoverflow.com/questions/74046734/plot-text-saliency-map-in-jupyter-notebook

    Parameters
    ----------
    explanations:
        List of tuples (tokens, salience) containing batch of explanations.

    Returns
    -------
    plot: matplotplib.pyplot.figure object, which will be automatically rendered by jupyter.
    """

    import matplotlib.pyplot as plt

    h_len = len(explanations)
    v_len = len(explanations[0][0])

    tokens = [i[0] for i in explanations]
    scores = [i[1] for i in explanations]

    fig, axes = plt.subplots(
        h_len,
        v_len,
        figsize=(v_len * 0.75, h_len * 1.25),
        gridspec_kw=dict(left=0.0, right=1.0),
    )
    plt.subplots_adjust(hspace=1.0, wspace=0.0)
    for i, ax in enumerate(axes):
        color_mapper = ColorMapper(np.max(scores[i]), np.min(scores[i]))
        if labels:
            ax[v_len // 2].set_title(labels[i])
        for j in range(v_len):
            color = color_mapper.to_rgb(scores[i][j], normalize_to_1=True)
            rect = plt.Rectangle((0, 0), 1, 1, color=color)
            ax[j].add_patch(rect)
            ax[j].text(0.5, 0.5, tokens[i][j], ha="center", va="center")
            ax[j].set_xlim(0, 1)
            ax[j].set_ylim(0, 1)
            ax[j].axis("off")
            ax[j] = fig.add_axes([0, 0.05, 1, 0.9], fc=[0, 0, 0, 0])

    ax = axes.ravel()[-1]
    for axis in ["left", "right"]:
        ax.spines[axis].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def plot_token_pruning_experiment(score: np.ndarray, original_prediction: np.ndarray):
    """
    AU-MSE - area under the mean squared error (y0−ymt)
    curve for pruning. Lower is better and indicates that removing less
    relevant nodes has little effect on the model prediction.
    """
    if len(score.shape) != 2:
        raise ValueError(f"Scores must be 2 dimensional.")

    y = np.asarray(list(range(len(score[0])))) + 1
    x = mse(
        np.broadcast_to(original_prediction, (score.shape[1], score.shape[0])).T, score
    )
    plt.plot(y, x)

    plt.title("Token Pruning Experiment")
    plt.xlabel("Number of tokens flipped")
    plt.ylabel("MSE")