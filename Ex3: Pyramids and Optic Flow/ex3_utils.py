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
    # Convert images to grayscale if they are not already
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY) if len(im1.shape) > 2 else im1
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY) if len(im2.shape) > 2 else im2

    # Compute derivatives in x and y directions
    I_X = cv2.filter2D(im2, -1, np.array([[1, 0, -1]]), borderType=cv2.BORDER_REPLICATE)
    I_Y = cv2.filter2D(im2, -1, np.array([[1], [0], [-1]]), borderType=cv2.BORDER_REPLICATE)
    I_T = im2 - im1

    # Initialize arrays to store velocities and corresponding points
    velocities = []
    points = []

    for i in range(step_size, im1.shape[0], step_size):
        for j in range(step_size, im1.shape[1], step_size):
            # Create a sample window for current point
            window_X = I_X[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1]
            window_Y = I_Y[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1]
            window_T = I_T[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1]

            # Flatten the window arrays
            window_X = window_X.flatten()
            window_Y = window_Y.flatten()
            window_T = window_T.flatten()

            # Calculate matrices
            A = np.array([[np.sum(window_X * window_X), np.sum(window_X * window_Y)],
                          [np.sum(window_X * window_Y), np.sum(window_Y * window_Y)]])
            B = np.array([[-np.sum(window_X * window_T)],
                          [-np.sum(window_Y * window_T)]])

            # Compute eigenvalues and eigenvectors
            eigen_val, eigen_vec = np.linalg.eig(A)
            eig_val1 = eigen_val[0]
            eig_val2 = eigen_val[1]

            # Ensure eigenvalues satisfy the conditions
            if eig_val1 < eig_val2:
                eig_val1, eig_val2 = eig_val2, eig_val1

            # Check conditions for valid optical flow
            if eig_val2 <= 1 or eig_val1 / eig_val2 >= 100:
                continue

            # Calculate u and v components of velocity
            u, v = np.linalg.inv(A) @ B

            points.append([j, i])
            velocities.append([u[0], v[0]])

    return np.array(points), np.array(velocities)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int, stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """

    img1_pyramid = gaussianPyr(img1, k)
    img2_pyramid = gaussianPyr(img2, k)

    # Optical flow calculation for the last pyramid level
    xy_prev, uv_prev = opticalFlow(img1_pyramid[-1], img2_pyramid[-1], stepSize, winSize)
    xy_prev = list(xy_prev)
    uv_prev = list(uv_prev)

    for i in range(1, k):
        # Calculate optical flow for the current level
        xy_i, uv_i = opticalFlow(img1_pyramid[-1 - i], img2_pyramid[-1 - i], stepSize, winSize)
        uv_i = list(uv_i)
        xy_i = list(xy_i)

        for g in range(len(xy_i)):
            xy_i[g] = list(xy_i[g])

        # Update uv according to the formula
        for j in range(len(xy_prev)):
            xy_prev[j] = [element * 2 for element in xy_prev[j]]
            uv_prev[j] = [element * 2 for element in uv_prev[j]]

        # If the locations of movements we found are new, append them; otherwise, add them to the proper location
        for j in range(len(xy_i)):
            if xy_i[j] in xy_prev:
                uv_prev[j] += uv_i[j]
            else:
                xy_prev.append(xy_i[j])
                uv_prev.append(uv_i[j])

    # Convert uv and xy to a 3-dimensional array
    flow_array = np.zeros(shape=(img1.shape[0], img1.shape[1], 2))

    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            if [y, x] not in xy_prev:
                flow_array[x, y] = [0, 0]
            else:
                flow_array[x, y] = uv_prev[xy_prev.index([y, x])]

    return flow_array


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    # If the images are not grayscale, convert them to grayscale
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Compute the derivative in the x-direction for the second image
    derivative_filter = np.array([[1, 0, -1]])
    I_X = cv2.filter2D(im2, -1, derivative_filter, borderType=cv2.BORDER_REPLICATE)

    # Compute the difference between the first and second image
    diff = im2 - im1

    # Compute the correlation matrix for the derivative and difference
    correlation = cv2.filter2D(I_X, -1, diff, borderType=cv2.BORDER_REPLICATE)

    # Compute the translation parameters using the Lucas-Kanade equation
    dx = np.sum(np.sum(I_X ** 2, axis=0) * np.sum(correlation, axis=0)) / np.sum(I_X ** 2)
    dy = np.sum(np.sum(I_X ** 2, axis=1) * np.sum(correlation, axis=1)) / np.sum(I_X ** 2)

    # Create the translation matrix
    translation_mat = np.array([[1, 0, dx],
                                [0, 1, dy],
                                [0, 0, 1]], dtype=np.float32)

    return translation_mat


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    # Convert images to grayscale if necessary
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Perform feature detection using goodFeaturesToTrack
    corners1 = cv2.goodFeaturesToTrack(im1, maxCorners=100, qualityLevel=0.3, minDistance=7)

    # Perform optical flow using calcOpticalFlowPyrLK
    corners2, status, _ = cv2.calcOpticalFlowPyrLK(im1, im2, corners1, None)

    # Filter out invalid points and corresponding corners
    valid_corners1 = corners1[status == 1]
    valid_corners2 = corners2[status == 1]

    # Estimate rigid transformation using estimateAffinePartial2D
    m, _ = cv2.estimateAffinePartial2D(valid_corners1, valid_corners2)

    # Convert the 2x3 transformation matrix to a 3x3 matrix
    rigid_matrix = np.zeros((3, 3), dtype=np.float32)
    rigid_matrix[:2, :] = m

    # Set the last row of the matrix to [0, 0, 1]
    rigid_matrix[2, 2] = 1.0

    return rigid_matrix


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """

    # Convert images to grayscale if necessary
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Compute the correlation using matchTemplate
    result = cv2.matchTemplate(im1, im2, cv2.TM_CCORR_NORMED)

    # Find the location of the best match
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Calculate the translation matrix
    translation_matrix = np.array([[1, 0, max_loc[0] - im1.shape[1]],
                                   [0, 1, max_loc[1] - im1.shape[0]]], dtype=np.float32)

    return translation_matrix


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """

    # Convert images to grayscale if necessary
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Compute the correlation using matchTemplate
    result = cv2.matchTemplate(im1, im2, cv2.TM_CCORR_NORMED)

    # Find the location of the best match
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Calculate the rigid matrix
    dx = max_loc[0] - im1.shape[1]
    dy = max_loc[1] - im1.shape[0]
    theta = 0  # Assuming no rotation for now

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rigid_matrix = np.array([[cos_theta, -sin_theta, dx],
                             [sin_theta, cos_theta, dy],
                             [0.0, 0.0, 1.0]], dtype=np.float32)

    return rigid_matrix


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    # Apply the transformation matrix T to image2 using cv2.warpPerspective
    warped_image = cv2.warpPerspective(im2, T, im1.shape[::-1])

    # Display both image1 and the warped image2 in the same figure
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(im1, cmap='gray')
    ax[0].set_title('Image 1')
    ax[1].imshow(warped_image, cmap='gray')
    ax[1].set_title('Warped Image 2')
    plt.show()

    return warped_image


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
    # Adjust the image size to match the pyramid levels
    img_height = np.power(2, levels) * int(img.shape[0] / np.power(2, levels))
    img_width = np.power(2, levels) * int(img.shape[1] / np.power(2, levels))
    img = img[0:img_height, 0:img_width]

    pyramid = [img]  # List to store pyramid images
    kernel_size = 5

    for i in range(1, levels):
        # Generate Gaussian kernel
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        kernel = cv2.getGaussianKernel(kernel_size, sigma)

        # Apply the kernel for blurring
        img = cv2.filter2D(img, -1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
        img = cv2.filter2D(img, -1, kernel=np.transpose(kernel), borderType=cv2.BORDER_REPLICATE)

        # Downsample the image by a factor of 2
        img = img[::2, ::2]

        # Add the downsampled image to the pyramid
        pyramid.append(img)

    return pyramid


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pyramid = []

    # Check if the image is already grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert the input image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)

    for _ in range(levels):
        # Apply Gaussian blur to the image
        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        # Expand the blurred image using pyrUp
        expanded = cv2.pyrUp(blurred)

        # Crop the expanded image to match the size of the original image
        height, width = img.shape[:2]
        expanded = expanded[:height, :width]

        # Compute the Laplacian by subtracting the expanded image from the original image
        laplacian = img - expanded

        # Add the Laplacian image to the pyramid
        pyramid.append(laplacian.astype(np.uint8))  # Convert back to np.uint8 for compatibility

        # Set the blurred image as the input for the next iteration
        img = blurred

    # Add the smallest level (Gaussian image) to the pyramid
    pyramid.append(img)

    return pyramid


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    # Get the number of levels in the Laplacian pyramid
    levels = len(lap_pyr)

    # Start with the smallest level (Gaussian image) from the Laplacian pyramid
    img = lap_pyr[levels - 1]

    # Iterate through the Laplacian pyramid in reverse order (from the second-smallest level to the largest)
    for i in range(levels - 2, -1, -1):
        # Expand the image to the size of the next level
        expanded = cv2.pyrUp(img)

        # Get the dimensions of the next level
        height, width = lap_pyr[i].shape[:2]

        # Crop the expanded image to match the size of the next level
        expanded = expanded[:height, :width]

        # Add the Laplacian of the current level to the expanded image
        img = expanded + lap_pyr[i]

    return img


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
    # Crop images and mask to a size compatible with the pyramid levels
    crop_size = np.power(2, levels)
    img_1 = crop_image(img_1, crop_size)
    img_2 = crop_image(img_2, crop_size)
    mask = crop_image(mask, crop_size)

    # Initialize the blended image
    im_blend = np.zeros(img_1.shape)

    # Check if the image is RGB
    if img_1.ndim > 2 or img_2.ndim > 2:
        # Perform blending for each channel separately
        for channel in range(3):
            part_im1 = img_1[:, :, channel]
            part_im2 = img_2[:, :, channel]
            part_mask = mask[:, :, channel]

            # Generate Gaussian pyramid for image 1
            gauss_pyr_1 = gaussianPyr(part_im1, levels)

            # Generate Gaussian pyramid for image 2
            gauss_pyr_2 = gaussianPyr(part_im2, levels)

            # Generate Laplacian pyramid for image 1
            lap_pyr_1 = laplaceianReduce(part_im1, levels)

            # Generate Laplacian pyramid for image 2
            lap_pyr_2 = laplaceianReduce(part_im2, levels)

            # Generate Gaussian pyramid for the mask
            gauss_pyr_mask = gaussianPyr(part_mask, levels)

            lp_ret = []
            for i in range(levels):
                # Ensure the images and mask have compatible dimensions at each pyramid level
                img_shape = gauss_pyr_mask[i].shape
                lap_pyr_1[i] = crop_image(lap_pyr_1[i], img_shape[1])
                lap_pyr_2[i] = crop_image(lap_pyr_2[i], img_shape[1])
                gauss_pyr_mask[i] = crop_image(gauss_pyr_mask[i], img_shape[1])

                curr_lap = gauss_pyr_mask[i] * lap_pyr_1[i] + (1 - gauss_pyr_mask[i]) * lap_pyr_2[i]
                lp_ret.append(curr_lap)
            im_blend[:, :, channel] = laplaceianExpand(lp_ret)

    else:
        # Perform blending for grayscale image
        # Generate Gaussian pyramid for image 1
        gauss_pyr_1 = gaussianPyr(img_1, levels)

        # Generate Gaussian pyramid for image 2
        gauss_pyr_2 = gaussianPyr(img_2, levels)

        # Generate Laplacian pyramid for image 1
        lap_pyr_1 = laplaceianReduce(img_1, levels)

        # Generate Laplacian pyramid for image 2
        lap_pyr_2 = laplaceianReduce(img_2, levels)

        # Generate Gaussian pyramid for the mask
        gauss_pyr_mask = gaussianPyr(mask, levels)

        lp_ret = []
        for i in range(levels):
            # Ensure the images and mask have compatible dimensions at each pyramid level
            img_shape = gauss_pyr_mask[i].shape
            lap_pyr_1[i] = crop_image(lap_pyr_1[i], img_shape[1])
            lap_pyr_2[i] = crop_image(lap_pyr_2[i], img_shape[1])
            gauss_pyr_mask[i] = crop_image(gauss_pyr_mask[i], img_shape[1])

            curr_lap = gauss_pyr_mask[i] * lap_pyr_1[i] + (1 - gauss_pyr_mask[i]) * lap_pyr_2[i]
            lp_ret.append(curr_lap)
        im_blend = laplaceianExpand(lp_ret)

    # Naive blend without using pyramid
    naive_blend = mask * img_1 + (1 - mask) * img_2

    return naive_blend, im_blend

def crop_image(img: np.ndarray, target_size: int) -> np.ndarray:
    height, width = img.shape[:2]
    new_height = min(height, target_size)
    new_width = min(width, target_size)
    start_height = (height - new_height) // 2
    start_width = (width - new_width) // 2
    return img[start_height:start_height+new_height, start_width:start_width+new_width]