from ..metrics import *
from .perturb_func import *
from .similar_func import *


AVAILABLE_METRICS = {
    "Faithfulness": {
        "Faithfulness Correlation": FaithfulnessCorrelation,
        "Faithfulness Estimate": FaithfulnessEstimate,
        "Pixel-Flipping": PixelFlipping,
        "Region Segmentation": RegionPerturbation,
        "Monotonicity-Arya": MonotonicityArya,
        "Monotonicity-Nguyen": MonotonicityNguyen,
        "Selectivity": Selectivity,
        "SensitivityN": SensitivityN,
        "IROF": IterativeRemovalOfFeatures,
        #"Infidelity": Infidelity,
    },
    "Robustness": {
        "Continuity Test": Continuity,
        "Local Lipschitz Estimate": LocalLipschitzEstimate,
        "Max-Sensitivity": MaxSensitivity,
        "Avg-Sensitivity": AvgSensitivity,
    },
    "Localisation": {
        "Pointing Game": PointingGame,
        "Top-K Intersection": TopKIntersection,
        "Relevance Mass Accuracy": RelevanceMassAccuracy,
        "Relevance Mass Ranking": RelevanceRankAccuracy,
        "Attribution Localisation ": AttributionLocalisation,
        "AUC": AUC,
    },
    "Complexity": {
        "Sparseness": Sparseness,
        "Complexity": Complexity,
        "Effective Complexity": EffectiveComplexity,
    },
    "Randomisation": {
        "Model Parameter Randomisation": ModelParameterRandomisation,
        "Random Logit": RandomLogit,
    },
    "Axiomatic": {
        "Completeness": Completeness,
        "NonSensitivity": NonSensitivity,
        "InputInvariance": InputInvariance,
    },
}


AVAILABLE_PERTURBATION_FUNCTIONS = {
    "gaussian_noise": gaussian_noise,
    "baseline_replacement_by_indices": baseline_replacement_by_indices,
    "baseline_replacement_by_patch": baseline_replacement_by_patch,
    "rotation": rotation,
    "translation_x_direction": translation_x_direction,
    "translation_y_direction": translation_y_direction,
    "uniform_sampling": uniform_sampling,
    "no_perturbation": no_perturbation,
}


AVAILABLE_SIMILARITY_FUNCTIONS = {
    "correlation_spearman": correlation_spearman,
    "correlation_pearson": correlation_pearson,
    "correlation_kendall_tau": correlation_kendall_tau,
    "distance_euclidean": distance_euclidean,
    "distance_manhattan": distance_manhattan,
    "distance_chebyshev": distance_chebyshev,
    "lipschitz_constant": lipschitz_constant,
    "abs_difference": abs_difference,
    "difference": difference,
    "cosine": cosine,
    "ssim": ssim,
    "mse": mse,
}

AVAILABLE_NORMALISATION_FUNCTIONS = {
    "normalise_by_negative": normalise_by_negative,
    "normalise_by_max": normalise_by_max,
    "denormalise": denormalise,
}


AVAILABLE_XAI_METHODS = {
    "Gradient",
    "Saliency",
    "GradientShap",
    "IntegratedGradients",
    "InputXGradient",
    "Occlusion",
    "FeatureAblation",
    "GradCam",
    "Control Var. Sobel Filter",
    "Control Var. Constant",
}


def available_categories() -> list:
    return [c for c in AVAILABLE_METRICS.keys()]


def available_metrics() -> dict:
    return {c: list(metrics.keys()) for c, metrics in AVAILABLE_METRICS.items()}


def available_methods() -> list:
    return [c for c in AVAILABLE_XAI_METHODS.keys()]


def available_perturbation_functions() -> list:
    return [c for c in AVAILABLE_PERTURBATION_FUNCTIONS.keys()]


def available_similarity_functions() -> list:
    return [c for c in AVAILABLE_SIMILARITY_FUNCTIONS.keys()]


def available_normalisation_functions() -> list:
    return [c for c in AVAILABLE_NORMALISATION_FUNCTIONS.keys()]
