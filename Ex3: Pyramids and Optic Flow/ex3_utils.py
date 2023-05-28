import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 313308785

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    # If the image is not grayscale, convert it to grayscale
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Compute the derivatives in x and y directions
    derivative_filter = np.array([[1, 0, -1]])
    I_X = cv2.filter2D(im2, -1, derivative_filter, borderType=cv2.BORDER_REPLICATE)
    I_Y = cv2.filter2D(im2, -1, derivative_filter.T, borderType=cv2.BORDER_REPLICATE)
    I_T = im2 - im1

    # Initialize arrays to store the calculated velocities and corresponding points
    velocities = []
    points = []

    for i in range(step_size, im1.shape[0], step_size):
        for j in range(step_size, im1.shape[1], step_size):
            # Create a small sample of I_X, I_Y, and I_T for the current window
            sample_I_X = I_X[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1]
            sample_I_Y = I_Y[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1]
            sample_I_T = I_T[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1]

            # Flatten the sample arrays since they will be treated as vectors
            sample_I_X = sample_I_X.flatten()
            sample_I_Y = sample_I_Y.flatten()
            sample_I_T = sample_I_T.flatten()

            # Size of the samples
            n = len(sample_I_X)

            # Calculate matrices (A^tA)^-1 and A^tB
            sum_IX_squared = sum(sample_I_X[h] ** 2 for h in range(n))
            sum_IX_IY = sum(sample_I_X[h] * sample_I_Y[h] for h in range(n))
            sum_IY_squared = sum(sample_I_Y[h] ** 2 for h in range(n))

            sum_IX_IT = sum(sample_I_X[h] * sample_I_T[h] for h in range(n))
            sum_IY_IT = sum(sample_I_Y[h] * sample_I_T[h] for h in range(n))

            # Enter the calculated values into a 2x2 matrix
            A = np.array([[sum_IX_squared, sum_IX_IY], [sum_IX_IY, sum_IY_squared]])
            B = np.array([[-sum_IX_IT], [-sum_IY_IT]])

            # Get eigenvalues and eigenvectors
            eigen_val, eigen_vec = np.linalg.eig(A)
            eig_val1 = eigen_val[0]
            eig_val2 = eigen_val[1]

            # Ensure eigenvalues satisfy the conditions
            if eig_val1 < eig_val2:
                temp = eig_val1
                eig_val1 = eig_val2
                eig_val2 = temp

            # Condition 2: If it holds, add the point
            if eig_val2 <= 1 or eig_val1 / eig_val2 >= 100:
                continue

            # Calculate u and v components of velocity
            vector_u_v = np.linalg.inv(A) @ B
            u = vector_u_v[0][0]
            v = vector_u_v[1][0]

            points.append([j, i])
            velocities.append([u, v])

    return np.array(points), np.array(velocities)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    img1_pyr = gaussianPyr(img1, k)
    img2_pyr = gaussianPyr(img2, k)

    xy_prev, uv_prev = opticalFlow(img1_pyr[-1], img2_pyr[-1], stepSize, winSize)
    xy_prev = xy_prev.tolist()
    uv_prev = uv_prev.tolist()

    for i in range(1, k):
        xy_i, uv_i = opticalFlow(img1_pyr[-1 - i], img2_pyr[-1 - i], stepSize, winSize)
        xy_i = xy_i.tolist()
        uv_i = uv_i.tolist()

        for j in range(len(xy_i)):
            xy_i[j] = xy_i[j].tolist()

        for j in range(len(xy_prev)):
            xy_prev[j] = [element * 2 for element in xy_prev[j]]
            uv_prev[j] = [element * 2 for element in uv_prev[j]]

        for j in range(len(xy_i)):
            if xy_i[j] in xy_prev:
                uv_prev[j] += uv_i[j]
            else:
                xy_prev.append(xy_i[j])
                uv_prev.append(uv_i[j])

    flow_matrix = np.zeros(shape=(img1.shape[0], img1.shape[1], 2))

    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            if [y, x] not in xy_prev:
                flow_matrix[x, y] = [0, 0]
            else:
                flow_matrix[x, y] = uv_prev[xy_prev.index([y, x])]

    if flow_matrix is None:
        return np.zeros(shape=(img1.shape[0], img1.shape[1], 2))  # Return empty flow matrix as fallback

    return flow_matrix


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    pass


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pass


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pass


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    pass


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    pass

