"""This modules holds a collection of perturbation functions i..e, ways to perturb an input or an explanation."""


def warn_normalise_abs(normalise: bool, abs: bool) -> None:
    print(
        f"Normalising attributions (or taking their absolute values) may destroy or skew information "
        f"in the explanation and as a result, affect the overall evaluation outcome. "
        f"\nNormalisation is set to {normalise} and absolute value is set to {abs}"
    )


def warn_noise_zero(noise: float) -> None:
    print(
        f"Noise is set to {noise:.2f} which is likely to invalidate the evaluation outcome of the test"
        f" given that it depends on perturbation of input(s)/ attribution(s). "
        f"\n Recommended to re-parameterise the metric."
    )
