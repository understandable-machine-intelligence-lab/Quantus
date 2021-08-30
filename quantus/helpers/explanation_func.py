import scipy
import random
import cv2
from captum.attr import *
from .utils import *

# from ..helpers.constants import XAI_METHODS
from typing import Union
import torch


def explain(
    model: torch.nn,
    inputs: Union[np.array, torch.Tensor],
    targets: Union[np.array, torch.Tensor],
    **kwargs,
) -> torch.Tensor:
    """
    Explain inputs given a model, targets and an explanation method.

    Expecting inputs to be shaped such as (batch_size, nr_channels, img_size, img_size)

    Returns np.ndarray of same shape as inputs.
    """

    if "explanation_func" not in kwargs:
        print_warning(
            warning_text=f"Using 'explain' function without specifying 'explanation_func' (str)"
            f"in kwargs will produce 'Gradient' explanation.\n"
        )

    explanation_func = kwargs.get("explanation_func", "Gradient").lower()

    # Set model in evaluate mode.
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
            .reshape(
                -1,
                kwargs.get("nr_channels", 3),
                kwargs.get("img_size", 224),
                kwargs.get("img_size", 224),
            )
            .to(kwargs.get("device", None))
        )
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).to(kwargs.get("device", None))

    explanation: torch.Tensor = torch.zeros_like(inputs)

    if explanation_func == "GradientShap".lower():
        explanation = (
            GradientShap(model)
            .attribute(
                inputs=inputs,
                target=targets,
                baselines=kwargs.get("baseline", torch.zeros_like(inputs)),
            )
            .sum(axis=1)
        )

    elif explanation_func == "IntegratedGradients".lower():
        explanation = (
            IntegratedGradients(model)
            .attribute(
                inputs=inputs,
                target=targets,
                baselines=kwargs.get("baseline", torch.zeros_like(inputs)),
                n_steps=10,
                method="riemann_trapezoid",
            )
            .sum(axis=1)
        )

    elif explanation_func == "InputXGradient".lower():
        explanation = (
            InputXGradient(model).attribute(inputs=inputs, target=targets).sum(axis=1)
        )

    elif explanation_func == "Saliency".lower():
        explanation = (
            Saliency(model)
            .attribute(inputs=inputs, target=targets, abs=True)
            .sum(axis=1)
        )

    elif explanation_func == "Gradient".lower():
        explanation = (
            Saliency(model)
            .attribute(inputs=inputs, target=targets, abs=False)
            .sum(axis=1)
        )

    elif explanation_func == "Occlusion".lower():

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

    elif explanation_func == "FeatureAblation".lower():

        explanation = (
            FeatureAblation(model).attribute(inputs=inputs, target=targets).sum(axis=1)
        )

    elif explanation_func == "GradCam".lower():

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

    elif explanation_func == "Control Var. Sobel Filter".lower():
        if len(explanation.shape) == 4:
            for i in range(len(explanation)):
                explanation[i] = torch.reshape(
                    input=torch.Tensor(
                        np.clip(
                            scipy.ndimage.sobel(inputs[i].cpu().numpy()), 0, 1
                        ).mean(axis=0)
                    ),
                    shape=(kwargs.get("img_size", 224), kwargs.get("img_size", 224)),
                )
        else:
            explanation = torch.reshape(
                input=torch.Tensor(
                    np.clip(scipy.ndimage.sobel(inputs.cpu().numpy()), 0, 1).mean(
                        axis=0
                    )
                ),
                shape=(kwargs.get("img_size", 224), kwargs.get("img_size", 224)),
            )

    elif explanation_func == "Control Var. Constant".lower():

        # TODO. So that the fill_dict is not calculating on batch rather than on an invidiual sample basis.
        assert (
            "fill_value" in kwargs
        ), "Specify a 'fill_value' e.g., 0.0 or 'black' for pixel replacement."

        fill_dict = {
            "random": float(random.random()),
            "uniform": float(random.uniform(explanation.min(), explanation.max())),
            "black": float(explanation.min()),
            "white": float(explanation.max()),
        }

        if isinstance(kwargs["fill_value"], (float, int)):
            fill_value = kwargs["fill_value"]
        else:
            fill_value = fill_dict[kwargs["fill_value"].lower()]

        explanation = torch.Tensor().new_full(
            size=explanation.shape, fill_value=fill_value
        )

    else:
        raise KeyError(
            "Specify a XAI method that already has been implemented {}."
        ).__format__("XAI_METHODS")

    if kwargs.get("abs", False):
        explanation = explanation.abs()

    if kwargs.get("pos_only", False):
        explanation[explanation < 0] = 0.0

    if kwargs.get("neg_only", False):
        explanation[explanation > 0] = 0.0

    if kwargs.get("normalize", True):
        explanation = normalize_heatmap(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            return explanation.cpu().detach().numpy()
        return explanation.cpu().numpy()
    return explanation
