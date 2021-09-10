from ..metrics import *
from .perturb_func import *
from .similar_func import *
from .local_func import *

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


AVAILABLE_METRICS = {
    "Faithfulness": {
        "Faithfulness Correlation": FaithfulnessCorrelation,
        "Faithfulness Estimate": FaithfulnessEstimate,
        "Pixel-Flipping": PixelFlipping,
        "Region Segmentation": RegionPerturbation,
        "Monotonicity Arya": MonotonicityArya,
        "Monotonicity Nguyen": MonotonicityNguyen,
        "Infidelity": Infidelity,
        "Selectivity": Selectivity,
        "SensitivityN": SensitivityN,
        "IROF": IROF,
    },
    "Robustness": {
        "Continuity Test": Continuity,
        "Input Independence Rate": InputIndependenceRate,
        "Local Lipschitz Estimate": LocalLipschitzEstimate,
        "Max-Sensitivity": MaxSensitivity,
        "Avg-Sensitivity": AvgSensitivity,
    },
    "Localisation": {
        "Pointing Game": PointingGame,
        "TKI": TopKIntersection,
        "Relevance Mass Accuracy": "ADD",
        "Relevance Mass Ranking": RelevanceRankAccuracy,
        "Attribution Localization ": AttributionLocalization,
    },
    "Complexity": {
        "Sparseness Test": Sparseness,
        "Complexity Test": Complexity,
        "Effective Complexity": EffectiveComplexity,
    },
    "Randomisation": {
        "Model Parameter Randomisation Test": "ADD",
        "Random Logit Test": "ADD",
    },
    "Axiomatic": {
        "Completeness Test": Completeness,
        "Symmetry": Symmetry,
        "InputInvariance": InputInvariance,
        "Sensitivity": Sensitivity,
        "Dummy": Dummy,
    },
}


AVAILABLE_PERTURBATION_FUNCTIONS = {
    "gaussian_blur": gaussian_blur,
    "gaussian_noise": gaussian_noise,
    "baseline_replacement_by_indices": baseline_replacement_by_indices,
    "baseline_replacement_by_patch": baseline_replacement_by_patch,
    "rotation": rotation,
    "translation_x_direction": translation_x_direction,
    "translation_y_direction": translation_y_direction,
    "optimization_scheme": optimization_scheme,
    "uniform_sampling": uniform_sampling,
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


AVAILABLE_LOCALIZATION_FUNCTIONS = {
    "localisation": localisation,
}


DEFAULT_METRICS = {
    "Faithfulness": FaithfulnessCorrelation(),
    "Max-Sensitivity": MaxSensitivity(),
    "Complexity": Complexity(),
}

DEFAULT_XAI_METHODS = ["Saliency", "IntegratedGradients"]
