from typing import Callable, Union
import torch
import torchvision
import cv2
from captum.attr import *
from .utils import *


def explain(model: torch.nn,
            inputs: Union[np.array, torch.Tensor],
            targets: Union[np.array, torch.Tensor],
            explanation_func: Callable,
            **kwargs) -> torch.Tensor:
    """
    Explain inputs given a model, targets and an explanation method.

    Expecting inputs to be shaped such as (batch_size, nr_channels, img_size, img_size)
    """

    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
            .reshape(
                -1,
                kwargs.get("nr_channels", 3),
                kwargs.get("img_size", 224),
                kwargs.get("img_size", 224),
            ).to(kwargs.get("device", None))
        )
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).to(kwargs.get("device", None))

    explanation: torch.Tensor = torch.zeros_like(inputs)

    if explanation_func == "GradientShap":
        explanation = (
            GradientShap(model)
            .attribute(
                inputs=inputs,
                target=targets,
                baselines=kwargs.get("baseline", torch.zeros_like(inputs)),
            )
            .sum(axis=1)
        )

    elif explanation_func == "IntegratedGradients":
        explanation = (
            IntegratedGradients(model)
            .attribute(
                inputs=inputs,
                target=targets,
                baselines=kwargs.get("baseline", torch.zeros_like(inputs)),
            )
            .sum(axis=1)
        )

    elif explanation_func == "InputXGradient":
        explanation = (
            InputXGradient(model).attribute(inputs=inputs, target=targets).sum(axis=1)
        )

    elif explanation_func == "Saliency":
        explanation = (
            Saliency(model)
            .attribute(inputs=inputs, target=targets, abs=True)
            .sum(axis=1)
        )

    elif explanation_func == "Gradient":
        explanation = (
            Saliency(model)
            .attribute(inputs=inputs, target=targets, abs=False)
            .sum(axis=1)
        )

    elif explanation_func == "Occlusion":

        assert (
            "sliding_window" in kwargs
        ), "Provide kwargs, 'oc_sliding_window' e.g., (4, 4) to compute an Occlusion explanation."

        explanation = (
            Occlusion(model)
            .attribute(
                inputs=inputs,
                target=targets,
                sliding_window_shapes=kwargs["oc_sliding_window"],
            )
            .sum(axis=1)
        )

    elif explanation_func == "FeatureAblation":

        explanation = (
            FeatureAblation(model).attribute(inputs=inputs, target=targets).sum(axis=1)
        )

    elif explanation_func == "GradCam":

        assert (
            "gc_layer" in kwargs
        ), "Provide kwargs, 'gc_layer' e.g., list(model.named_modules())[1][1][-6] to run GradCam."

        explanation = (
            LayerGradCam(model, layer=kwargs["gc_layer"])
            .attribute(inputs=inputs, target=targets)
            .sum(axis=1)
        )
        explanation = torch.Tensor(
            cv2.resize(
                explanation.cpu().data.numpy(),
                dsize=(kwargs.get("img_size", 224), kwargs.get("img_size", 224)),
            )
        )

    else:
        raise KeyError("Specify a XAI method that exists.")

    if kwargs.get("abs", False):
        explanation = explanation.abs()

    if kwargs.get("pos_only", False):
        explanation[explanation < 0] = 0.0

    if kwargs.get("neg_only", False):
        explanation[explanation > 0] = 0.0

    # if kwargs.get("normalise", False):
    #   explanation = normalize_heatmap(explanation)

    return explanation
