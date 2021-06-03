""" Collection of perturbation functions i..e, ways to perturb an input or an explanation. """
import numpy as np
import scipy


def gaussian_blur(a, **kwargs):
    """Inject gaussian blur to the input. """
    return scipy.ndimage.gaussian_filter(a, sigma=0.02 * np.max(a))
