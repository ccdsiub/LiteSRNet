# -*- coding: utf-8 -*-

"""Peak Signal-to-Noise Ratio (PSNR).

This code is adapted from https://github.com/Booooooooooo/

This script contains the following functions:
    * calculate_psnr: Calculate PSNR.
"""

__author__ = "Mir Sazzat Hossain"


import math

import numpy as np


def calculate_psnr(
    sr: np.ndarray,
    hr: np.ndarray,
    rgb_range: float,
) -> float:
    """Calculate PSNR.

    :param sr: super-resulotion image
    :type sr: np.ndarray
    :param hr: high-resolution image
    :type hr: np.ndarray
    :param rgb_range: maximum value of RGB
    :type rgb_range: int

    :return: PSNR
    :rtype: float
    """
    mse = np.mean((hr - sr) ** 2)
    if (mse == 0):
        return 100
    psnr = 20 * math.log10(rgb_range / math.sqrt(mse))
    return psnr
