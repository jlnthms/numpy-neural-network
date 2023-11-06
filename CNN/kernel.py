from typing import Tuple

import numpy as np


class Kernel:
    def __init__(self, size: Tuple[int, int]):
        self.size = size
        self.array = np.zeros(size)

    def convolve(self, image: np.ndarray, stride=1, padding=0):
        kh, kw = self.array.shape
        ih, iw, num_channels = image.shape
        ph, pw = padding, padding
        output_h = (ih - kh + 2 * ph) // stride + 1
        output_w = (iw - kw + 2 * pw) // stride + 1
        feature_map = np.zeros((output_h, output_w, num_channels), dtype=image.dtype)

        for c in range(num_channels):
            for i in range(0, ih - kh + 1, stride):
                for j in range(0, iw - kw + 1, stride):
                    region = image[i:i + kh, j:j + kw, c]
                    feature_map[i // stride, j // stride, c] = np.sum(self.array * region)

        return feature_map