""" Collection of perturbation methods. """
import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_blur(data):
    """ Adds gaussian blur to the input. """

    return gaussian_filter(data, sigma=0.02 * np.max(data))
