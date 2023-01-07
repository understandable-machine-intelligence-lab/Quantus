from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from quantus.nlp.helpers.types import Explanation


def create_div(explanation: Explanation, label: str):
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
    max_score = np.max(scores)

    for t, g in zip(tokens, scores):
        # Calculate color based on relevance score in explanation.
        green = 255.0 - 255.0 * (g / max_score)
        blue = 255.0 - 255.0 * (g / max_score)
        token_span = token_span_template.replace(
            "{{color}}", f"rgb(255,{green},{blue})"
        ).replace("{{token}}", t)
        body += token_span + " "

    return div_template.replace("{{label}}", label).replace("{{saliency_map}}", body)


def visualise_explanations_as_html(
    explanations: List[Explanation], labels: Optional[List[str]]
) -> str:
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
        label = labels[i] if labels is not None else None
        div = create_div(explanation, label)
        spans += div
    return heatmap_template.replace("{{body}}", spans)


def visualise_explanations_as_pyplot(explanations: List[Explanation]) -> plt:

    """
    Plots attributions over a batch of text sequence explanations.

    References:
        - https://stackoverflow.com/questions/74046734/plot-text-saliency-map-in-jupyter-notebook

    Parameters
    ----------
    explanations: List of Named tuples (tokens, salience) containing batch of explanations.

    Returns
    -------
    plot: matplotplib.pyplot object, which will be automatically rendered by jupyter.
    """

    h_len = len(explanations)
    v_len = len(explanations[0][0])

    tokens = np.asarray([i[0] for i in explanations]).reshape(-1)
    colors = np.asarray([i[1] for i in explanations]).reshape(-1)

    fig, axes = plt.subplots(
        h_len,
        v_len,
        figsize=(v_len, h_len * 0.5),
        gridspec_kw=dict(left=0.0, right=1.0),
    )
    for i, ax in enumerate(axes.ravel()):
        rect = plt.Rectangle((0, 0), 1, 1, color=(1.0, 1 - colors[i], 1 - colors[i]))
        ax.add_patch(rect)
        ax.text(0.5, 0.5, tokens[i], ha="center", va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax = fig.add_axes([0, 0.05, 1, 0.9], fc=[0, 0, 0, 0])
    for axis in ["left", "right"]:
        ax.spines[axis].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return plt
