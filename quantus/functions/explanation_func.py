"""This modules contains explainer functions which can be used in conjunction with the metrics in the library."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import warnings
from importlib import util
from typing import Optional, Union

import numpy as np
import scipy

from quantus.helpers import constants
from quantus.helpers import __EXTRAS__
from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers.utils import (
    get_baseline_value,
    infer_channel_first,
    make_channel_last,
)


if util.find_spec("torch"):
    import torch
if util.find_spec("captum"):
    from captum.attr import (
        GradientShap,
        IntegratedGradients,
        InputXGradient,
        Saliency,
        Occlusion,
        FeatureAblation,
        LayerGradCam,
        DeepLift,
        DeepLiftShap,
        GuidedGradCam,
        Deconvolution,
        FeaturePermutation,
        Lime,
        KernelShap,
        LRP,
        LayerConductance,
        LayerActivation,
        InternalInfluence,
        LayerGradientXActivation,
    )
if util.find_spec("zennit"):
    from zennit import canonizers as zcanon
    from zennit import composites as zcomp
    from zennit import attribution as zattr
    from zennit import core as zcore
if util.find_spec("tensorflow"):
    import tensorflow as tf
if util.find_spec("tf_explain"):
    import tf_explain


def explain(model, inputs, targets, **kwargs) -> np.ndarray:
    """
    Explain inputs given a model, targets and an explanation method.
    Expecting inputs to be shaped such as (batch_size, nr_channels, ...) or (batch_size, ..., nr_channels).

    Parameters
    ----------
    model: torch.nn.Module, tf.keras.Model
            A model that is used for explanation.
    inputs: np.ndarray
             The inputs that ought to be explained.
    targets: np.ndarray
             The target lables that should be used in the explanation.
    kwargs: optional
            Keyword arguments.

    Returns
    -------
    explanation: np.ndarray
             Returns np.ndarray of same shape as inputs.
    """

    if util.find_spec("captum") or util.find_spec("tf_explain"):
        if "method" not in kwargs:
            warnings.warn(
                f"Using quantus 'explain' function as an explainer without specifying 'method' (string) "
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

    Parameters
    ----------
    model: torch.nn.Module, tf.keras.Model
            A model that is used for explanation.
    inputs: np.ndarray
         The inputs that ought to be explained.
    targets: np.ndarray
         The target lables that should be used in the explanation.
    kwargs: optional
            Keyword arguments.

    Returns
    -------
    explanation: np.ndarray
         Returns np.ndarray of same shape as inputs.
    """
    xai_lib = kwargs.get("xai_lib", "captum")
    if isinstance(model, torch.nn.Module):
        if util.find_spec("captum") and util.find_spec("zennit"):
            if xai_lib == "captum":
                return generate_captum_explanation(model, inputs, targets, **kwargs)
            if xai_lib == "zennit":
                return generate_zennit_explanation(model, inputs, targets, **kwargs)
        if util.find_spec("captum"):
            return generate_captum_explanation(model, inputs, targets, **kwargs)
        if util.find_spec("zennit"):
            return generate_zennit_explanation(model, inputs, targets, **kwargs)
    if isinstance(model, tf.keras.Model):
        if util.find_spec("tf_explain"):
            return generate_tf_explanation(model, inputs, targets, **kwargs)
        else:
            raise ValueError(
                f"Model is of type tf.keras.Model but tf_explain is not installed."
            )

    raise ValueError(
        f"Model needs to be tf.keras.Model or torch.nn.Module but is {type(model)}. "
        "Please install Captum or Zennit for torch>=1.2 models and tf-explain for TensorFlow>=2.0."
    )


def generate_tf_explanation(
    model, inputs: np.array, targets: np.array, **kwargs
) -> np.ndarray:
    """
    Generate explanation for a tf model with tf_explain.
    Assumption: Currently only normalised absolute values of explanations supported.

    Parameters
    ----------
    model: tf.keras.Model
            A model that is used for explanation.
    inputs: np.ndarray
         The inputs that ought to be explained.
    targets: np.ndarray
         The target lables that should be used in the explanation.
    kwargs: optional
            Keyword arguments.

    Returns
    -------
    explanation: np.ndarray
         Returns np.ndarray of same shape as inputs.

    """
    method = kwargs.get("method", "VanillaGradients")
    method_kwargs = kwargs.get("method_kwargs", {})
    inputs = inputs.reshape(-1, *model.input_shape[1:])
    if not isinstance(targets, np.ndarray):
        targets = np.array([targets])

    channel_first = kwargs.get("channel_first", infer_channel_first(inputs))
    inputs = make_channel_last(inputs, channel_first)

    explanation: np.ndarray = np.zeros_like(inputs)

    if method in constants.DEPRECATED_XAI_METHODS_TF:
        warnings.warn(
            f"Explanaiton method string {method} is deprecated. Use "
            f"{constants.DEPRECATED_XAI_METHODS_TF[method]} instead.\n",
            category=UserWarning,
        )
        method = constants.DEPRECATED_XAI_METHODS_TF[method]

    if method == "VanillaGradients":
        explainer = tf_explain.core.vanilla_gradients.VanillaGradients()
        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: explainer.explain(
                            ([x], None), model, y, **method_kwargs
                        ),
                        inputs,
                        targets,
                    )
                ),
                dtype=float,
            )
            / 255
        )

    elif method == "IntegratedGradients":
        n_steps = kwargs.get("n_steps", 10)
        explainer = tf_explain.core.integrated_gradients.IntegratedGradients()
        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: explainer.explain(
                            ([x], None), model, y, n_steps=n_steps, **method_kwargs
                        ),
                        inputs,
                        targets,
                    )
                ),
                dtype=float,
            )
            / 255
        )

    elif method == "GradientsInput":
        explainer = tf_explain.core.gradients_inputs.GradientsInputs()
        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: explainer.explain(
                            ([x], None), model, y, **method_kwargs
                        ),
                        inputs,
                        targets,
                    )
                ),
                dtype=float,
            )
            / 255
        )

    elif method == "OcclusionSensitivity":
        patch_size = kwargs.get("window", (1, *([4] * (inputs.ndim - 2))))[-1]
        explainer = tf_explain.core.occlusion_sensitivity.OcclusionSensitivity()
        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: explainer.explain(
                            ([x], None),
                            model,
                            y,
                            patch_size=patch_size,
                            **method_kwargs,
                        ),
                        inputs,
                        targets,
                    )
                ),
                dtype=float,
            )
            / 255
        )

    elif method == "GradCAM":
        if "gc_layer" in kwargs:
            method_kwargs["layer_name"] = kwargs["gc_layer"]

        explainer = tf_explain.core.grad_cam.GradCAM()
        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: explainer.explain(
                            ([x], None), model, y, **method_kwargs
                        ),
                        inputs,
                        targets,
                    )
                ),
                dtype=float,
            )
            / 255
        )

    elif method == "SmoothGrad":

        num_samples = kwargs.get("num_samples", 5)
        noise = kwargs.get("noise", 0.1)
        explainer = tf_explain.core.smoothgrad.SmoothGrad()
        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: explainer.explain(
                            ([x], None),
                            model,
                            y,
                            num_samples=num_samples,
                            noise=noise,
                            **method_kwargs,
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
            f"Specify a XAI method that already has been implemented {constants.AVAILABLE_XAI_METHODS_TF}."
        )

    if (
        not kwargs.get("normalise", True)
        or not kwargs.get("abs", True)
        or not kwargs.get("pos_only", True)
        or kwargs.get("neg_only", False)
        or kwargs.get("reduce_axes", None) is not None
    ):
        raise KeyError(
            "Only normalizsd absolute explanations are currently supported for TensorFlow models (tf-explain). "
            "Set normalise=true, abs=true, pos_only=true, neg_only=false. reduce_axes parameter is not available; "
            "explanations are always reduced over channel axis."
        )

    return explanation


def generate_captum_explanation(
    model,
    inputs: np.ndarray,
    targets: np.ndarray,
    device: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """
    Generate explanation for a torch model with captum.
    Parameters
    ----------
    model: torch.nn.Module
        A model that is used for explanation.
    inputs: np.ndarray
         The inputs that ought to be explained.
    targets: np.ndarray
         The target lables that should be used in the explanation.
    device: string
        Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
    kwargs: optional
            Keyword arguments. May include method_kwargs dictionary which includes keyword arguments for a method call.

    Returns
    -------
    explanation: np.ndarray
         Returns np.ndarray of same shape as inputs.
    """

    method = kwargs.get("method", "Gradient")
    method_kwargs = kwargs.get("method_kwargs", {})

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

    # Prevent attribution summation for 2D-data. Recreate np.sum behavior when passing reduce_axes=(), i.e. no change.
    if (len(tuple(kwargs.get("reduce_axes", [1]))) == 0) | (inputs.ndim < 3):

        def f_reduce_axes(a):
            return a

    else:

        def f_reduce_axes(a):
            return a.sum(**reduce_axes)

    explanation: torch.Tensor = torch.zeros_like(inputs)

    if method in constants.DEPRECATED_XAI_METHODS_CAPTUM:
        warnings.warn(
            f"Explanaiton method string {method} is deprecated. Use "
            f"{constants.DEPRECATED_XAI_METHODS_CAPTUM[method]} instead.\n",
            category=UserWarning,
        )
        method = constants.DEPRECATED_XAI_METHODS_CAPTUM[method]

    if method in ["GradientShap", "DeepLift", "DeepLiftShap"]:
        attr_func = eval(method)
        explanation = f_reduce_axes(
            attr_func(model, **method_kwargs).attribute(
                inputs=inputs,
                target=targets,
                baselines=kwargs.get("baseline", torch.zeros_like(inputs)),
            )
        )

    elif method == "IntegratedGradients":
        attr_func = eval(method)
        explanation = f_reduce_axes(
            attr_func(model, **method_kwargs).attribute(
                inputs=inputs,
                target=targets,
                baselines=kwargs.get("baseline", torch.zeros_like(inputs)),
                n_steps=10,
                method="riemann_trapezoid",
            )
        )

    elif method in [
        "InputXGradient",
        "Saliency",
        "FeatureAblation",
        "Deconvolution",
        "FeaturePermutation",
        "Lime",
        "KernelShap",
        "LRP",
    ]:
        attr_func = eval(method)
        explanation = f_reduce_axes(
            attr_func(model, **method_kwargs).attribute(inputs=inputs, target=targets)
        )

    elif method == "Gradient":
        explanation = f_reduce_axes(
            Saliency(model, **method_kwargs).attribute(
                inputs=inputs, target=targets, abs=False
            )
        )

    elif method == "Occlusion":
        window_shape = kwargs.get("window", (1, *([4] * (inputs.ndim - 2))))
        explanation = f_reduce_axes(
            Occlusion(model).attribute(
                inputs=inputs,
                target=targets,
                sliding_window_shapes=window_shape,
            )
        )

    elif method in [
        "LayerGradCam",
        "GuidedGradCam",
        "LayerConductance",
        "LayerActivation",
        "InternalInfluence",
        "LayerGradientXActivation",
    ]:
        if "gc_layer" in kwargs:
            method_kwargs["layer"] = kwargs["gc_layer"]

        if "layer" not in method_kwargs:
            raise ValueError(
                "Specify a convolutional layer name as 'gc_layer' to run GradCam."
            )

        if isinstance(method_kwargs["layer"], str):
            method_kwargs["layer"] = eval(method_kwargs["layer"])

        attr_func = eval(method)

        if method != "LayerActivation":
            explanation = attr_func(model, **method_kwargs).attribute(
                inputs=inputs, target=targets
            )
        else:
            explanation = attr_func(model, **method_kwargs).attribute(inputs=inputs)

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
        else:
            if explanation.shape[-1] != inputs.shape[-1]:
                warnings.warn(
                    "Quantus requires GradCam attribution and input to correspond in "
                    "last dimensions, but got shapes {} and {}\n "
                    "Pass 'interpolate' argument to explanation function get matching dimensions.".format(
                        explanation.shape, inputs.shape
                    ),
                    category=UserWarning,
                )

        explanation = f_reduce_axes(explanation)

    elif method == "Control Var. Sobel Filter":
        explanation = torch.zeros(size=inputs.shape)

        for i in range(len(explanation)):
            explanation[i] = torch.Tensor(
                np.clip(scipy.ndimage.sobel(inputs[i].cpu().numpy()), 0, 1)
            )
        explanation = explanation.mean(**reduce_axes)

    elif method == "Control Var. Random Uniform":
        explanation = torch.rand(size=(inputs.shape[0], *inputs.shape[2:]))

    elif method == "Control Var. Constant":
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
            f"Specify a XAI method that already has been implemented {constants.AVAILABLE_XAI_METHODS_CAPTUM}."
        )

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            explanation = explanation.cpu().detach().numpy()
        else:
            explanation = explanation.cpu().numpy()

    return explanation


def generate_zennit_explanation(
    model,
    inputs: np.ndarray,
    targets: np.ndarray,
    device: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """
    Generate explanation for a torch model with zennit.

    Parameters
    ----------
    model: torch.nn.Module
        A model that is used for explanation.
    inputs: np.ndarray
         The inputs that ought to be explained.
    targets: np.ndarray
         The target lables that should be used in the explanation.
    device: string
        Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
    kwargs: optional
            Keyword arguments.

    Returns
    -------
    explanation: np.ndarray
         Returns np.ndarray of same shape as inputs.

    """

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
            "The specified composite {} is not valid. "
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

    return explanation
