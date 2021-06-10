import torch
import cv2
from captum.attr import *
from .utils import *


def explain(model, inputs, targets, xai_method, device=None, **params) -> torch.Tensor:
    """
    Explain inputs given a model, targets and an explanation method.

    (batch_size, nr_channels, img_size, img_size) where batch_size = 1, 256
    """

    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
            .reshape(
                -1,
                params.get("nr_channels", 3),
                params.get("img_size", 224),
                params.get("img_size", 224),
            )
            .to(device)
        )
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).to(
            device
        )  # torch.nn.functional.one_hot(torch.Tensor(targets)).to(device) #torch.Tensor(targets).to(device)

    explanation: torch.Tensor = torch.zeros_like(inputs)

    if xai_method == "GradientShap":
        explanation = (
            GradientShap(model)
            .attribute(
                inputs=inputs,
                target=targets,
                baselines=params.get("baseline", torch.zeros_like(inputs)),
            )
            .sum(axis=1)
        )

    elif xai_method == "IntegratedGradients":
        explanation = (
            IntegratedGradients(model)
            .attribute(
                inputs=inputs,
                target=targets,
                baselines=params.get("baseline", torch.zeros_like(inputs)),
            )
            .sum(axis=1)
        )

    elif xai_method == "InputXGradient":
        explanation = (
            InputXGradient(model).attribute(inputs=inputs, target=targets).sum(axis=1)
        )

    elif xai_method == "Saliency":
        explanation = (
            Saliency(model)
            .attribute(inputs=inputs, target=targets, abs=True)
            .sum(axis=1)
        )

    elif xai_method == "Gradient":
        explanation = (
            Saliency(model)
            .attribute(inputs=inputs, target=targets, abs=False)
            .sum(axis=1)
        )

    elif xai_method == "Occlusion":

        assert (
            "sliding_window" in params
        ), "Provide params, 'oc_sliding_window' e.g., (4, 4) to compute an Occlusion explanation."

        explanation = (
            Occlusion(model)
            .attribute(
                inputs=inputs,
                target=targets,
                sliding_window_shapes=params["oc_sliding_window"],
            )
            .sum(axis=1)
        )

    elif xai_method == "FeatureAblation":

        explanation = (
            FeatureAblation(model).attribute(inputs=inputs, target=targets).sum(axis=1)
        )

    elif xai_method == "GradCam":

        assert (
            "gc_layer" in params
        ), "Provide params, 'gc_layer' e.g., list(model.named_modules())[1][1][-6] to run GradCam."

        explanation = (
            LayerGradCam(model, layer=params["gc_layer"])
            .attribute(inputs=inputs, target=targets)
            .sum(axis=1)
        )
        explanation = torch.Tensor(
            cv2.resize(
                explanation.cpu().data.numpy(),
                dsize=(params.get("img_size", 224), params.get("img_size", 224)),
            )
        )

    else:
        raise KeyError("Specify a XAI method that exists.")

    if params.get("abs", False):
        explanation = explanation.abs()

    if params.get("pos_only", False):
        explanation[explanation < 0] = 0.0

    if params.get("neg_only", False):
        explanation[explanation > 0] = 0.0

    # if params.get("normalise", False):
    #   explanation = normalize_heatmap(explanation)

    return explanation
