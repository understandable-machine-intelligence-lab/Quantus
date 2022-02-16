"""This modules contains explainer functions which can be used in conjunction with the metrics in the library."""
from typing import Union

import numpy as np
import scipy
import random
from importlib import util
import cv2
import warnings
from .utils import *
from .normalise_func import *
from ..helpers import __EXTRAS__

if util.find_spec("captum"):
    from captum.attr import *
if util.find_spec("zennit"):
    from zennit import canonizers as zcanon
    from zennit import composites as zcomp
    from zennit import attribution as zattr
    from zennit import core as zcore
if util.find_spec("tf_explain"):
    import tf_explain


def explain(model, inputs, targets, *args, **kwargs) -> np.ndarray:
    """
    Explain inputs given a model, targets and an explanation method.

    Expecting inputs to be shaped such as (batch_size, nr_channels, img_size, img_size) or (batch_size, img_size, img_size, nr_channels).

    Returns np.ndarray of same shape as inputs.
    """

    if util.find_spec("captum") or util.find_spec("tf_explain"):
        if "method" not in kwargs:
            warnings.warn(
                f"Using quantus 'explain' function as an explainer without specifying 'method' (str) "
                f"in kwargs will produce a vanilla 'Gradient' explanation.\n",
                category=UserWarning,
            )
    if util.find_spec("zennit"):
        if "attributor" not in kwargs:
            warnings.warn(
                f"Using quantus 'explain' function as an explainer without specifying 'attributor'"
                f"in kwargs will produce a vanilla 'Gradient' explanation.\n",
                category=UserWarning,
            )

    if not __EXTRAS__:
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
    model: tf.keras.Model, inputs: np.array, targets: np.array, **kwargs
) -> np.ndarray:
    """
    Generate explanation for a tf model with tf_explain.
    Currently only normalised absolute values of explanations supported.
    """
    method = kwargs.get("method", "Gradient").lower()
    inputs = inputs.reshape(-1, *model.input_shape[1:])

    channel_first = kwargs.get("channel_first", get_channel_first(inputs))
    inputs = get_channel_last_batch(inputs, channel_first)

    explanation: np.ndarray = np.zeros_like(inputs)

    if method == "Gradient".lower():
        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: tf_explain.core.vanilla_gradients.VanillaGradients().explain(
                            ([x], None), model, y
                        ),
                        inputs,
                        targets,
                    )
                ),
                dtype=float,
            )
            / 255
        )

    elif method == "IntegratedGradients".lower():
        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: tf_explain.core.integrated_gradients.IntegratedGradients().explain(
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
        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: tf_explain.core.gradients_inputs.GradientsInputs().explain(
                            ([x], None), model, y
                        ),
                        inputs,
                        targets,
                    )
                ),
                dtype=float,
            )
            / 255
        )

    elif method == "Occlusion".lower():
        patch_size = kwargs.get("window", (1, 4, 4))[-1]
        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: tf_explain.core.occlusion_sensitivity.OcclusionSensitivity().explain(
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
        assert (
            "gc_layer" in kwargs
        ), "Specify convolutional layer name as 'gc_layer' to run GradCam."

        explanation = (
            np.array(
                list(
                    map(
                        lambda x, y: tf_explain.core.grad_cam.GradCAM().explain(
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
            "Specify a XAI method that already has been implemented {}."
        ).__format__("XAI_METHODS")

    if (
        not kwargs.get("normalise", True)
        or not kwargs.get("abs", True)
        or not kwargs.get("pos_only", True)
        or kwargs.get("neg_only", False)
    ):
        raise KeyError(
            "Only normalized absolute explanations are currently supported for TensorFlow models (tf-explain). "
            "Set normalise=true, abs=true, pos_only=true, neg_only=false."
        )

    return explanation


def generate_captum_explanation(
    model: ModelInterface,
    inputs: np.ndarray,
    targets: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Generate explanation for a torch model with captum."""
    method = kwargs.get("method", "Gradient").lower()
    # Set model in evaluate mode.
    model.to(kwargs.get("device", None))
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.Tensor(inputs).to(kwargs.get("device", None))

    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).to(kwargs.get("device", None))

    inputs = inputs.reshape(
        -1,
        kwargs.get("nr_channels", 3),
        kwargs.get("img_size", 224),
        kwargs.get("img_size", 224),
    )

    explanation: torch.Tensor = torch.zeros_like(inputs)

    if method == "GradientShap".lower():
        explanation = (
            GradientShap(model)
            .attribute(
                inputs=inputs,
                target=targets,
                baselines=kwargs.get("baseline", torch.zeros_like(inputs)),
            )
            .sum(axis=1)
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
            .sum(axis=1)
        )

    elif method == "InputXGradient".lower():
        explanation = (
            InputXGradient(model).attribute(inputs=inputs, target=targets).sum(axis=1)
        )

    elif method == "Saliency".lower():
        explanation = (
            Saliency(model)
            .attribute(inputs=inputs, target=targets, abs=True)
            .sum(axis=1)
        )

    elif method == "Gradient".lower():
        explanation = (
            Saliency(model)
            .attribute(inputs=inputs, target=targets, abs=False)
            .sum(axis=1)
        )

    elif method == "Occlusion".lower():
        explanation = (
            Occlusion(model)
            .attribute(
                inputs=inputs,
                target=targets,
                sliding_window_shapes=kwargs.get("window", (1, 4, 4)),
            )
            .sum(axis=1)
        )

    elif method == "FeatureAblation".lower():
        explanation = (
            FeatureAblation(model).attribute(inputs=inputs, target=targets).sum(axis=1)
        )

    elif method == "GradCam".lower():
        assert (
            "gc_layer" in kwargs
        ), "Provide kwargs, 'gc_layer' e.g., list(model.named_modules())[1][1][-6] to run GradCam."

        if isinstance(kwargs["gc_layer"], str):
            kwargs["gc_layer"] = eval(kwargs["gc_layer"])

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

    elif method == "Control Var. Sobel Filter".lower():
        explanation = torch.zeros(
            size=(inputs.shape[0], inputs.shape[2], inputs.shape[3])
        )

        for i in range(len(explanation)):
            explanation[i] = torch.Tensor(
                np.clip(scipy.ndimage.sobel(inputs[i].cpu().numpy()), 0, 1)
                .mean(axis=0)
                .reshape(kwargs.get("img_size", 224), kwargs.get("img_size", 224))
            )

    elif method == "Control Var. Constant".lower():
        assert (
            "constant_value" in kwargs
        ), "Specify a 'constant_value' e.g., 0.0 or 'black' for pixel replacement."

        explanation = torch.zeros(
            size=(inputs.shape[0], inputs.shape[2], inputs.shape[3])
        )

        # Update the tensor with values per input x.
        for i in range(explanation.shape[0]):
            constant_value = get_baseline_value(
                choice=kwargs["constant_value"], img=inputs[i]
            )
            explanation[i] = torch.Tensor().new_full(
                size=explanation[0].shape, fill_value=constant_value
            )

    else:
        raise KeyError(
            "Specify a XAI method that already has been implemented {}."
        ).__format__("XAI_METHODS")

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
    **kwargs,
) -> np.ndarray:
    """Generate explanation for a torch model with zennit."""
    # Get zennit composite, canonizer, attributor
    # Handle canonizer kwarg
    canonizer = kwargs.get("canonizer", None)
    if not canonizer == None and not issubclass(canonizer, zcanon.Canonizer):
        raise ValueError(
            "The specified canonizer is not valid. "
            "Please provide None or an instance of zennit.canonizers.Canonizer"
        )

    # Handle attributor kwarg
    # TODO: we could create a str --> attributor mapping, but I like this better
    attributor = kwargs.get("attributor", zattr.Gradient)
    if not issubclass(attributor, zattr.Attributor):
        raise ValueError(
            "The specified attributor is not valid. "
            "Please provide a subclass of zennit.attributon.Attributor"
        )

    # Handle composite kwarg
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
    model.to(kwargs.get("device", None))
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.Tensor(inputs).to(kwargs.get("device", None))

    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).to(kwargs.get("device", None))

    inputs = inputs.reshape(
        -1,
        kwargs.get("nr_channels", 3),
        kwargs.get("img_size", 224),
        kwargs.get("img_size", 224),
    )

    # Initialize canonizer, composite, and attributor
    if canonizer is not None:
        canonizers = [canonizer()]
    else:
        canonizers = []
    if composite is not None:
        # TODO: only uses default parameters for each method for now
        composite = composite(canonizers=canonizers)
    # TODO: only uses default parameters for each method for now
    attributor = attributor(model, composite)

    # TODO: there may be a better solution here?
    n_outputs = model(inputs).shape[1]

    # Get Attributions
    with attributor:

        # TODO: this assumes one-hot encoded target outputs (e.g., initial relevance).
        #  Better solution with more choices?
        eye = torch.eye(n_outputs, device=kwargs.get("device", None))
        output_target = eye[targets]
        output_target = output_target.reshape(-1, n_outputs)
        # print(inputs.shape, targets.shape, output_target.shape)
        _, explanation = attributor(inputs, output_target)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            explanation = explanation.cpu().detach().numpy()
        else:
            explanation = explanation.cpu().numpy()

    # TODO: Include alternatives here?
    # Remove channel axis
    explanation = np.sum(explanation, axis=1)

    if kwargs.get("normalise", False):
        explanation = kwargs.get("normalise_func", normalise_by_negative)(explanation)

    if kwargs.get("abs", False):
        explanation = np.abs(explanation)

    elif kwargs.get("pos_only", False):
        explanation[explanation < 0] = 0.0

    elif kwargs.get("neg_only", False):
        explanation[explanation > 0] = 0.0

    return explanation
