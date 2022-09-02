"""This modules contains explainer functions which can be used in conjunction with the metrics in the library."""
from typing import Dict, Optional, Union

import numpy as np
import scipy
from importlib import util
import cv2
import warnings
from .utils import *
from .normalise_func import *
from ..helpers import __EXTRAS__
from ..helpers import constants

if util.find_spec("torch"):
    import torch
if util.find_spec("captum"):
    from captum.attr import *
if util.find_spec("zennit"):
    from zennit import canonizers as zcanon
    from zennit import composites as zcomp
    from zennit import attribution as zattr
    from zennit import core as zcore
if util.find_spec("tensorflow"):
    import tensorflow as tf
if util.find_spec("tf_explain"):
    import tf_explain

from ..helpers import __EXTRAS__
from .model_interface import ModelInterface
from .normalise_func import normalise_by_negative
from .utils import get_baseline_value, infer_channel_first, make_channel_last


def explain(model, inputs, targets, **kwargs) -> np.ndarray:
    """
    Explain inputs given a model, targets and an explanation method.

    Expecting inputs to be shaped such as (batch_size, nr_channels, ...) or (batch_size, ..., nr_channels).

    Returns np.ndarray of same shape as inputs.
    """

    if util.find_spec("captum") or util.find_spec("tf_explain"):
        if "method" not in kwargs:
            warnings.warn(
                f"Using quantus 'explain' function as an explainer without specifying 'method' (str) "
                f"in kwargs will produce a vanilla 'Gradient' explanation.\n",
                category=UserWarning,
            )
    elif util.find_spec("zennit"):
        if "attributor" not in kwargs:
            warnings.warn(
                f"Using quantus 'explain' function as an explainer without specifying 'attributor'"
                f"in kwargs will produce a vanilla 'Gradient' explanation.\n",
                category=UserWarning,
            )

    elif not __EXTRAS__:
        raise ImportError(
            "Explanation library not found. Please install Captum or Zennit for torch>=1.2 models "
            "and tf-explain for TensorFlow>=2.0."
        )

    explanation = get_explanation(model, inputs, targets, **kwargs)

    return explanation


def get_explanation(model, inputs, targets, **kwargs):
    """
    Generate explanation array based on the type of input model and user specifications.
    For tensorflow models, tf.explain is used.
    For pytorch models, either captum or zennit is used, depending on which module is installed.
        If both are installed, captum is used per default. Setting the xai_lib kwarg to "zennit" uses zennit instead.
    """
    xai_lib = kwargs.get("xai_lib", "captum")
    if isinstance(model, torch.nn.modules.module.Module):
        if util.find_spec("captum") and util.find_spec("zennit"):
            if xai_lib == "captum":
                return generate_captum_explanation(model, inputs, targets, **kwargs)
            if xai_lib == "zennit":
                return generate_zennit_explanation(model, inputs, targets, **kwargs)
        if util.find_spec("captum"):
            return generate_captum_explanation(model, inputs, targets, **kwargs)
        if util.find_spec("zennit"):
            return generate_zennit_explanation(model, inputs, targets, **kwargs)
    if isinstance(model, tf.keras.Model) and util.find_spec("tf_explain"):
        return generate_tf_explanation(model, inputs, targets, **kwargs)

    raise ValueError(
        "Model needs to be tf.keras.Model or torch.nn.modules.module.Module. "
        "Please install Captum or Zennit for torch>=1.2 models and tf-explain for TensorFlow>=2.0."
    )


def generate_tf_explanation(
    model: ModelInterface, inputs: np.array, targets: np.array, **kwargs
) -> np.ndarray:
    """
    Generate explanation for a tf model with tf_explain.
    Currently only normalised absolute values of explanations supported.
    """
    method = kwargs.get("method", "Gradient").lower()
    inputs = inputs.reshape(-1, *model.input_shape[1:])
    if not isinstance(targets, np.ndarray):
        targets = np.array([targets])

    channel_first = kwargs.get("channel_first", infer_channel_first(inputs))
    inputs = make_channel_last(inputs, channel_first)

    explanation: np.ndarray = np.zeros_like(inputs)

    if method == "Gradient".lower():
        explainer = tf_explain.core.vanilla_gradients.VanillaGradients()
        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: explainer.explain(([x], None), model, y),
                        inputs,
                        targets,
                    )
                ),
                dtype=float,
            )
            / 255
        )

    elif method == "IntegratedGradients".lower():
        explainer = tf_explain.core.integrated_gradients.IntegratedGradients()
        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: explainer.explain(
                            ([x], None), model, y, n_steps=10
                        ),
                        inputs,
                        targets,
                    )
                ),
                dtype=float,
            )
            / 255
        )

    elif method == "InputXGradient".lower():
        explainer = tf_explain.core.gradients_inputs.GradientsInputs()
        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: explainer.explain(([x], None), model, y),
                        inputs,
                        targets,
                    )
                ),
                dtype=float,
            )
            / 255
        )

    elif method == "Occlusion".lower():
        patch_size = kwargs.get("window", (1, *([4] * (inputs.ndim - 2))))[-1]
        explainer = tf_explain.core.occlusion_sensitivity.OcclusionSensitivity()
        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: explainer.explain(
                            ([x], None), model, y, patch_size=patch_size
                        ),
                        inputs,
                        targets,
                    )
                ),
                dtype=float,
            )
            / 255
        )

    elif method == "GradCam".lower():
        if "gc_layer" not in kwargs:
            raise ValueError(
                "Specify a convolutional layer name as 'gc_layer' to run GradCam."
            )

        explainer = tf_explain.core.grad_cam.GradCAM()
        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: explainer.explain(
                            ([x], None), model, y, layer_name=kwargs["gc_layer"]
                        ),
                        inputs,
                        targets,
                    )
                ),
                dtype=float,
            )
            / 255
        )

    else:
        raise KeyError(
            f"Specify a XAI method that already has been implemented {constants.AVAILABLE_XAI_METHODS}."
        )

    if (
        not kwargs.get("normalise", True)
        or not kwargs.get("abs", True)
        or not kwargs.get("pos_only", True)
        or kwargs.get("neg_only", False)
        or kwargs.get("reduce_axes", None) is not None
    ):
        raise KeyError(
            "Only normalized absolute explanations are currently supported for TensorFlow models (tf-explain). "
            "Set normalise=true, abs=true, pos_only=true, neg_only=false. reduce_axes parameter is not available; "
            "explanations are always reduced over channel axis."
        )

    return explanation


def generate_captum_explanation(
    model: ModelInterface,
    inputs: np.ndarray,
    targets: np.ndarray,
    device: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """Generate explanation for a torch model with captum."""

    method = kwargs.get("method", "Gradient").lower()

    # Set model in evaluate mode.
    model.to(device)
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.Tensor(inputs).to(device)

    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).to(device)

    assert 0 not in kwargs.get(
        "reduce_axes", [1]
    ), "Reduction over batch_axis is not available, please do not include axis 0 in 'reduce_axes' kwargs."
    assert len(kwargs.get("reduce_axes", [1])) <= inputs.ndim - 1, (
        "Cannot reduce attributions over more axes than each sample has dimensions, but got "
        "{} and  {}.".format(len(kwargs.get("reduce_axes", [1])), inputs.ndim - 1)
    )

    reduce_axes = {"axis": tuple(kwargs.get("reduce_axes", [1])), "keepdims": True}

    explanation: torch.Tensor = torch.zeros_like(inputs)

    if method == "GradientShap".lower():
        explanation = (
            GradientShap(model)
            .attribute(
                inputs=inputs,
                target=targets,
                baselines=kwargs.get("baseline", torch.zeros_like(inputs)),
            )
            .sum(**reduce_axes)
        )

    elif method == "IntegratedGradients".lower():
        explanation = (
            IntegratedGradients(model)
            .attribute(
                inputs=inputs,
                target=targets,
                baselines=kwargs.get("baseline", torch.zeros_like(inputs)),
                n_steps=10,
                method="riemann_trapezoid",
            )
            .sum(**reduce_axes)
        )

    elif method == "InputXGradient".lower():
        explanation = (
            InputXGradient(model)
            .attribute(inputs=inputs, target=targets)
            .sum(**reduce_axes)
        )

    elif method == "Saliency".lower():
        explanation = (
            Saliency(model)
            .attribute(inputs=inputs, target=targets, abs=True)
            .sum(**reduce_axes)
        )

    elif method == "Gradient".lower():
        explanation = (
            Saliency(model)
            .attribute(inputs=inputs, target=targets, abs=False)
            .sum(**reduce_axes)
        )

    elif method == "Occlusion".lower():
        window_shape = kwargs.get("window", (1, *([4] * (inputs.ndim - 2))))
        explanation = (
            Occlusion(model)
            .attribute(
                inputs=inputs,
                target=targets,
                sliding_window_shapes=window_shape,
            )
            .sum(**reduce_axes)
        )

    elif method == "FeatureAblation".lower():
        explanation = (
            FeatureAblation(model)
            .attribute(inputs=inputs, target=targets)
            .sum(**reduce_axes)
        )

    elif method == "GradCam".lower():
        if "gc_layer" not in kwargs:
            raise ValueError(
                "Provide kwargs, 'gc_layer' e.g., list(model.named_modules())[-4][1] to run GradCam."
            )

        if isinstance(kwargs["gc_layer"], str):
            kwargs["gc_layer"] = eval(kwargs["gc_layer"])

        explanation = (
            LayerGradCam(model, layer=kwargs["gc_layer"])
            .attribute(inputs=inputs, target=targets)
            .sum(**reduce_axes)
        )
        if "interpolate" in kwargs:
            if isinstance(kwargs["interpolate"], tuple):
                if "interpolate_mode" in kwargs:
                    explanation = LayerGradCam.interpolate(
                        explanation,
                        kwargs["interpolate"],
                        interpolate_mode=kwargs["interpolate_mode"],
                    )
                else:
                    explanation = LayerGradCam.interpolate(
                        explanation, kwargs["interpolate"]
                    )

    elif method == "Control Var. Sobel Filter".lower():
        explanation = torch.zeros(size=inputs.shape)

        for i in range(len(explanation)):
            explanation[i] = torch.Tensor(
                np.clip(scipy.ndimage.sobel(inputs[i].cpu().numpy()), 0, 1)
            )
        explanation = explanation.mean(**reduce_axes)

    elif method == "Control Var. Random Uniform".lower():
        explanation = torch.rand(size=(inputs.shape[0], *inputs.shape[2:]))

    elif method == "Control Var. Constant".lower():
        assert (
            "constant_value" in kwargs
        ), "Specify a 'constant_value' e.g., 0.0 or 'black' for pixel replacement."

        explanation = torch.zeros(size=inputs.shape)

        # Update the tensor with values per input x.
        for i in range(explanation.shape[0]):
            constant_value = get_baseline_value(
                value=kwargs["constant_value"], arr=inputs[i], return_shape=(1,)
            )[0]
            explanation[i] = torch.Tensor().new_full(
                size=explanation[0].shape, fill_value=constant_value
            )

        explanation = explanation.mean(**reduce_axes)

    else:
        raise KeyError(
            f"Specify a XAI method that already has been implemented {constants.AVAILABLE_XAI_METHODS}."
        )

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            explanation = explanation.cpu().detach().numpy()
        else:
            explanation = explanation.cpu().numpy()

    if kwargs.get("normalise", False):
        explanation = kwargs.get("normalise_func", normalise_by_negative)(explanation)

    if kwargs.get("abs", False):
        explanation = np.abs(explanation)

    elif kwargs.get("pos_only", False):
        explanation[explanation < 0] = 0.0

    elif kwargs.get("neg_only", False):
        explanation[explanation > 0] = 0.0

    return explanation


def generate_zennit_explanation(
    model: ModelInterface,
    inputs: np.ndarray,
    targets: np.ndarray,
    device: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """Generate explanation for a torch model with zennit."""

    assert 0 not in kwargs.get(
        "reduce_axes", [1]
    ), "Reduction over batch_axis is not available, please do not include axis 0 in 'reduce_axes' kwarg."
    assert len(kwargs.get("reduce_axes", [1])) <= inputs.ndim - 1, (
        "Cannot reduce attributions over more axes than each sample has dimensions, but got "
        "{} and  {}.".format(len(kwargs.get("reduce_axes", [1])), inputs.ndim - 1)
    )

    reduce_axes = {"axis": tuple(kwargs.get("reduce_axes", [1])), "keepdims": True}

    # Get zennit composite, canonizer, attributor and handle canonizer kwargs.
    canonizer = kwargs.get("canonizer", None)
    if not canonizer == None and not issubclass(canonizer, zcanon.Canonizer):
        raise ValueError(
            "The specified canonizer is not valid. "
            "Please provide None or an instance of zennit.canonizers.Canonizer"
        )

    # Handle attributor kwargs.
    attributor = kwargs.get("attributor", zattr.Gradient)
    if not issubclass(attributor, zattr.Attributor):
        raise ValueError(
            "The specified attributor is not valid. "
            "Please provide a subclass of zennit.attributon.Attributor"
        )

    # Handle attributor kwargs.
    composite = kwargs.get("composite", None)
    if not composite == None and isinstance(composite, str):
        if composite not in zcomp.COMPOSITES.keys():
            raise ValueError(
                "Composite {} does not exist in zennit."
                "Please provide None, a subclass of zennit.core.Composite, or one of {}".format(
                    composite, zcomp.COMPOSITES.keys()
                )
            )
        else:
            composite = zcomp.COMPOSITES[composite]
    if not composite == None and not issubclass(composite, zcore.Composite):
        raise ValueError(
            "The specified composite is not valid. "
            "Please provide None, a subclass of zennit.core.Composite, or one of {}".format(
                composite, zcomp.COMPOSITES.keys()
            )
        )

    # Set model in evaluate mode.
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.Tensor(inputs).to(device)

    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).to(device)

    canonizer_kwargs = kwargs.get("canonizer_kwargs", {})
    composite_kwargs = kwargs.get("composite_kwargs", {})
    attributor_kwargs = kwargs.get("attributor_kwargs", {})

    # Initialize canonizer, composite, and attributor.
    if canonizer is not None:
        canonizers = [canonizer(**canonizer_kwargs)]
    else:
        canonizers = []
    if composite is not None:
        composite = composite(
            **{
                **composite_kwargs,
                "canonizers": canonizers,
            }
        )
    attributor = attributor(
        **{
            **attributor_kwargs,
            "model": model,
            "composite": composite,
        }
    )

    n_outputs = model(inputs).shape[1]

    # Get the attributions.
    with attributor:
        if "attr_output" in attributor_kwargs.keys():
            _, explanation = attributor(inputs, None)
        else:
            eye = torch.eye(n_outputs, device=device)
            output_target = eye[targets]
            output_target = output_target.reshape(-1, n_outputs)
            _, explanation = attributor(inputs, output_target)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            explanation = explanation.cpu().detach().numpy()
        else:
            explanation = explanation.cpu().numpy()

    # Sum over the axes.
    explanation = np.sum(explanation, **reduce_axes)

    if kwargs.get("normalise", False):
        explanation = kwargs.get("normalise_func", normalise_by_negative)(explanation)

    if kwargs.get("abs", False):
        explanation = np.abs(explanation)

    elif kwargs.get("pos_only", False):
        explanation[explanation < 0] = 0.0

    elif kwargs.get("neg_only", False):
        explanation[explanation > 0] = 0.0

    return explanation
