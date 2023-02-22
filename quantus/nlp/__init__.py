from __future__ import annotations


from quantus.nlp.helpers.constants import (
    available_categories,
    available_metrics,
    available_normalisation_functions,
    available_perturbation_functions,
    available_plain_text_perturbation_functions,
    available_latent_space_perturbation_functions,
    available_xai_methods,
    available_plain_text_xai_methods,
    available_numerical_xai_methods,
    AVAILABLE_PERTURBATION_FUNCTIONS,
)
from quantus.nlp.functions.explanation_func import explain
from quantus.nlp.functions.perturb_func import (
    spelling_replacement,
    synonym_replacement,
    typo_replacement,
    uniform_noise,
    gaussian_noise,
)

from quantus.nlp.functions.plot_func import (
    visualise_explanations_as_pyplot,
    visualise_explanations_as_html,
)

from quantus.nlp.helpers.model.text_classifier import TextClassifier, Tokenizer
from quantus.nlp.helpers.model.huggingface_tokenizer import HuggingFaceTokenizer
from importlib import util

if util.find_spec("tensorflow"):
    from quantus.nlp.helpers.model.tensorflow_huggingface_text_classifier import (
        TFHuggingFaceTextClassifier,
    )
if util.find_spec("torch"):
    from quantus.nlp.helpers.model.torch_huggingface_text_classifier import (
        TorchHuggingFaceTextClassifier,
    )

from quantus.nlp.metrics.robustness.avg_sensitivity import AvgSensitivity
from quantus.nlp.metrics.robustness.max_sensitivity import MaxSensitivity
from quantus.nlp.metrics.robustness.relative_input_stability import (
    RelativeInputStability,
)
from quantus.nlp.metrics.robustness.relative_output_stability import (
    RelativeOutputStability,
)
from quantus.nlp.metrics.robustness.relative_representation_stability import (
    RelativeRepresentationStability,
)
from quantus.nlp.metrics.robustness.local_lipschitz_estimate import (
    LocalLipschitzEstimate,
)
from quantus.nlp.metrics.randomisation.model_parameter_randomisation import (
    ModelParameterRandomisation,
)
from quantus.nlp.metrics.randomisation.random_logit import RandomLogit
from quantus.nlp.metrics.faithfullness.token_flipping import TokenFlipping

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantus.nlp.helpers.types import TF_TensorLike, TensorLike # pragma: not covered

from quantus.nlp.helpers.types import (
    Explanation,
    ExplainFn,
    PlainTextPerturbFn,
    NumericalPerturbFn,
    NormaliseFn,
    SimilarityFn,
    PerturbationType,
    NoiseType,
)

from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.helpers.utils import normalise_attributions, abs_attributions
from quantus.nlp.evaluation import evaluate
