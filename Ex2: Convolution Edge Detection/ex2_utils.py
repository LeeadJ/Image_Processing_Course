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
    # Define the convolution kernels for x and y derivatives
    kernel = np.array([[1, 0, -1]])

    # Compute x-derivative using convolution with the kernel
    G_X = cv2.filter2D(in_image, -1, kernel)
    # Compute y-derivative using convolution with the transposed kernel
    G_Y = cv2.filter2D(in_image, -1, kernel.T)

    # Compute magnitude of the gradient using element-wise operations
    magnitude = np.sqrt(G_X ** 2 + G_Y ** 2).astype(np.float64)

    # Compute direction of the gradient using element-wise arctangent
    direction = np.arctan2(G_Y, G_X).astype(np.float64)

    # Return the direction and magnitude matrices as a tuple
    return direction, magnitude


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    # Create a 1D Gaussian kernel using binomial coefficients
    kernel_1d = np.array([1, 1], dtype=np.float32)
    for _ in range(k_size - 2):
        kernel_1d = np.convolve(kernel_1d, [1, 1])
    kernel_1d = kernel_1d.reshape(1, -1)  # Convert to row vector

    # Create a 2D Gaussian kernel by convolving the 1D kernel with its transpose
    kernel_2d = np.matmul(kernel_1d.T, kernel_1d)

    # Normalize the kernel
    kernel_2d /= np.sum(kernel_2d)

    # Perform convolution between the image and the Gaussian kernel
    blurred_image = conv2D(in_image, kernel_2d)

    return blurred_image


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    # Create a 2D Gaussian kernel using OpenCV's built-in function
    kernel_2d = cv2.getGaussianKernel(k_size, -1)
    kernel_2d = np.matmul(kernel_2d, kernel_2d.T)

    # Perform convolution between the image and the Gaussian kernel using OpenCV's filter2D function
    blurred_image = cv2.filter2D(in_image, -1, kernel_2d, borderType=cv2.BORDER_REPLICATE)

    return blurred_image


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    # Apply Laplacian filter
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    # Find zero crossings
    rows, cols = laplacian.shape
    edge_matrix = np.zeros_like(laplacian, dtype=np.uint8)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = [laplacian[i - 1, j], laplacian[i + 1, j], laplacian[i, j - 1], laplacian[i, j + 1]]
            if np.any(np.diff(np.sign(neighbors))):
                edge_matrix[i, j] = 255

    return edge_matrix


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    # Apply Laplacian filter
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    # Find zero crossings
    rows, cols = laplacian.shape
    edge_matrix = np.zeros_like(laplacian, dtype=np.uint8)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = [laplacian[i - 1, j], laplacian[i + 1, j], laplacian[i, j - 1], laplacian[i, j + 1]]
            if np.any(np.diff(np.sign(neighbors))):
                edge_matrix[i, j] = 255

    return edge_matrix


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use Open CV function: cv.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Apply Hough circles transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30,
                               minRadius=min_radius, maxRadius=max_radius)

    # Process the detected circles
    detected_circles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for circle in circles:
            x, y, radius = circle
            detected_circles.append((x, y, radius))

    return detected_circles


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
