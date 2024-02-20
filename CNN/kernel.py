"""
Kernel

This module contains the base Kernel class

Author: Julien THOMAS
Date: November 04, 2023
"""

from typing import Tuple

import numpy as np


class Kernel:
    """
    A kernel is a filter, e.g a matrix of a defined size that is used to extract
    features at different scales on an image.

    Attributes:
        size (Tuple[int, int]): height and width of the kernel.
        array (np.ndarray): a matrix of shape 'size'.

    Methods:
        convolve(image, stride, padding): convolution of the kernel with an image.
    """
    def __init__(self, size: Tuple[int, int]):
        self.size = size
        self.array = np.zeros(size)

    import numpy as np

    def convolve(self, image: np.ndarray, stride=1, padding=0):
        """
        Kernel convolution over an image. The filter strides over the image and
        element-wise multiplication is applied at every iteration between the kernel
        array and the region of the image it is over.

        Args:
            image (np.ndarray): input image to be convolved with the kernel.
            stride (int): number of pixels the kernel is shifted each iteration.
            padding (int): number of lines added to each dimension in case kernel size does not divide image size.

        Returns:
             feature_map (np.ndarray): the input image after being convolved with the kernel.
        """
        kh, kw = self.array.shape
        if len(image.shape) == 3:
            ih, iw, num_channels = image.shape
        else:
            ih, iw = image.shape
            num_channels = 1
        ph, pw = padding, padding
        output_h = (ih - kh + 2 * ph) // stride + 1
        output_w = (iw - kw + 2 * pw) // stride + 1

        if num_channels == 1:
            feature_map = np.zeros((output_h, output_w), dtype=image.dtype)
        else:
            feature_map = np.zeros((output_h, output_w, num_channels), dtype=image.dtype)

        for i in range(0, ih - kh + 1, stride):
            for j in range(0, iw - kw + 1, stride):
                if num_channels == 1:
                    region = image[i:i + kh, j:j + kw]
                    feature_map[i // stride, j // stride] = np.sum(np.multiply(self.array, region))
                else:
                    region = image[i:i + kh, j:j + kw, :]
                    feature_map[i // stride, j // stride, :] = np.sum(np.multiply(self.array, region), axis=(0, 1))

        return feature_map

