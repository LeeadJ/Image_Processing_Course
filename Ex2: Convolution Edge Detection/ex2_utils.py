import math
import numpy as np
import cv2

def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """

    return 313308785

def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    # flip the kernel
    flipped_kernel = np.flip(k_size)

    # pad the kernel with zeros
    padded_kernel = np.pad(flipped_kernel, (len(in_signal) - 1, len(in_signal) - 1), mode='constant')

    # create vector to return
    vector_return = np.zeros(len(in_signal) + len(flipped_kernel) - 1)

    # preform the convolution
    for i in range(len(vector_return)):
        for j in range(len(in_signal)):
            vector_return[i] += in_signal[-1 - j] * padded_kernel[-1 - i + j]

    return vector_return


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    # Compute the dimensions of the input image and the kernel
    rows, cols = in_image.shape
    k_rows, k_cols = kernel.shape

    # Compute the padding required to ensure that the output image has the same size as the input image
    pad_rows = k_rows // 2
    pad_cols = k_cols // 2

    # Create a padded version of the input image
    padded_image = np.zeros((rows + 2 * pad_rows, cols + 2 * pad_cols))
    padded_image[pad_rows: pad_rows + rows, pad_cols: pad_cols + cols] = in_image

    # Create an output image
    out_image = np.zeros((rows, cols))

    # Convolve the input image with the kernel
    for i in range(pad_rows, rows + pad_rows):
        for j in range(pad_cols, cols + pad_cols):
            sub_image = padded_image[i - pad_rows: i + pad_rows + 1, j - pad_cols: j + pad_cols + 1]
            out_image[i - pad_rows, j - pad_cols] = np.sum(sub_image * kernel)

    return out_image


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """

    return


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    return


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    return
