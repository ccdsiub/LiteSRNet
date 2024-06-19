# -*- coding: utf-8 -*-

"""Structural Similarity (SSIM).

This code is adapted from https://github.com/Booooooooooo/

This script contains the following functions:
    * calculate_ssim: Calculate SSIM.
"""

__author__ = "Mir Sazzat Hossain"


import numpy as np
import torch
from scipy import signal


def trans2Y(image: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image to YCbCr image, then extract Y channel.

    :param image: RGB image
    :type image: torch.Tensor

    :return: Y channel of YCbCr image
    :rtype: torch.Tensor
    """
    image_r = image[:, 0, :, :]
    image_g = image[:, 1, :, :]
    image_b = image[:, 2, :, :]
    image_y = 0.256789 * image_r + 0.504129 * image_g + 0.097906 * image_b + 16
    return image_y


def matlab_style_gauss2D(
    shape: tuple = (3, 3),
    sigma: float = 0.5
) -> np.ndarray:
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial.

    :param shape: shape of the mask
    :type shape: tuple
    :param sigma: sigma of the Gaussian function
    :type sigma: float

    :return: 2D Gaussian mask
    :rtype: numpy.ndarray
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def calculate_ssim(
    X: torch.Tensor,
    Y: torch.Tensor,
    scale: int,
    dataset_type=None,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    R: float = 255,
) -> float:
    """
    SSIM index calculation (adapted from Matlab's implementation).

    :param X: y channel (i.e., luminance) of transformed YCbCr space of X
    :type X: torch.Tensor
    :param Y: y channel (i.e., luminance) of transformed YCbCr space of Y
    :type Y: torch.Tensor
    :param scale: scale of the image
    :type scale: int
    :param dataset: dataset name
    :type dataset: str, optional
    :param sigma: sigma of the Gaussian filter
    :type sigma: float, optional
    :param K1: parameter K1 of SSIM
    :type K1: float, optional
    :param K2: parameter K2 of SSIM
    :type K2: float, optional
    :param R: dynamic range of the image (255 for the 8-bit grayscale images)
    :type R: int, optional

    :return: SSIM index
    :rtype: float
    """
    gaussian_filter = matlab_style_gauss2D((11, 11), sigma)

    X = trans2Y(X).squeeze()
    Y = trans2Y(Y).squeeze()
    X = X.cpu().numpy().astype(np.float64)
    Y = Y.cpu().numpy().astype(np.float64)

    shave = scale
    if dataset and not dataset.dataset.benchmark:
        shave = scale + 6
    X = X[shave:-shave, shave:-shave]
    Y = Y[shave:-shave, shave:-shave]

    window = gaussian_filter / np.sum(np.sum(gaussian_filter))

    window = np.fliplr(window)
    window = np.flipud(window)

    ux = signal.convolve2d(X, window, mode='valid',
                           boundary='fill', fillvalue=0)
    uy = signal.convolve2d(Y, window, mode='valid',
                           boundary='fill', fillvalue=0)

    uxx = signal.convolve2d(X * X, window, mode='valid',
                            boundary='fill', fillvalue=0)
    uyy = signal.convolve2d(Y * Y, window, mode='valid',
                            boundary='fill', fillvalue=0)
    uxy = signal.convolve2d(X * Y, window, mode='valid',
                            boundary='fill', fillvalue=0)

    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy

    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1, 2 * vxy + C2,
                      ux ** 2 + uy ** 2 + C1, vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D
    mssim = S.mean()

    return mssim
