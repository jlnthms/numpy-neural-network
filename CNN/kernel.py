import numpy as np


class Kernel:
    def __init__(self, array: np.ndarray):
        self.array = array

    def shape(self):
        return self.array.shape if self.array else None

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
                    region = image[i:i+kh, j:j+kw, c]
                    feature_map[i//stride, j//stride, c] = np.sum(self.array * region)

        return feature_map


class GaussianKernel(Kernel):
    def __init__(self, sigma, size):
        array = np.array([])
        super().__init__(array)
        self.sigma = sigma
        self.size = size

        # Create the Gaussian kernel array
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * self.sigma ** 2)) * np.exp(
                -((x - self.size // 2) ** 2 + (y - self.size // 2) ** 2) / (2 * self.sigma ** 2)),
            (self.size, self.size)
        )
        self.array = kernel / np.sum(kernel)  # Normalize the kernel


class EdgeDetectionKernel(Kernel):
    def __init__(self):
        # Define the kernel elements for edge detection
        kernel_elements = np.array([[-1, -1, -1],
                                    [-1, 8, -1],
                                    [-1, -1, -1]], dtype=np.int32)
        super().__init__(kernel_elements)
