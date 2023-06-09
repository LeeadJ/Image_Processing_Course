from ex3_utils import *
import time


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("LK Demo - START")

    # Load and preprocess images
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=0.5, fy=0.5)

    # Apply perspective transformation to create img_2
    transformation_matrix = np.array([[1, 0, -0.2],
                                      [0, 1, -0.1],
                                      [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, transformation_matrix, img_1.shape[::-1])

    # Calculate optical flow
    start_time = time.time()
    points, optical_flow = opticalFlow(img_1.astype(np.float), img_2.astype(np.float), step_size=20, win_size=5)
    end_time = time.time()

    # Print time taken and flow statistics
    print("Time: {:.4f}".format(end_time - start_time))
    print(np.median(optical_flow, 0))
    print(np.mean(optical_flow, 0))

    # Display optical flow
    displayOpticalFlow(img_2, points, optical_flow)
    print("LK Demo - END")


def hierarchicalkDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Hierarchical LK Demo - START")

    # Load and preprocess the images
    img1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)

    # Apply a perspective transformation to create the second image
    transformation_matrix = np.array([[1, 0, -0.2],
                                      [0, 1, -0.1],
                                      [0, 0, 1]], dtype=np.float)
    img2 = cv2.warpPerspective(img1, transformation_matrix, img1.shape[::-1])

    # Perform hierarchical Lucas-Kanade optical flow
    start_time = time.time()
    optical_flow_result = opticalFlowPyrLK(img1.astype(np.float), img2.astype(np.float), k=5, stepSize=20,
                                              winSize=5)
    points, velocities = convert_3d_array_to_points(optical_flow_result)
    end_time = time.time()

    # Print timing information and statistics of optical flow velocities
    print("Time: {:.4f}".format(end_time - start_time))
    print("Median velocity:", np.median(velocities, axis=0))
    print("Mean velocity:", np.mean(velocities, axis=0))

    # Display the optical flow
    displayOpticalFlow(img2, points, velocities)

    print("Hierarchical LK Demo - END")


def compareLK(img_path):
    """
    ADD TEST
    Compare the two results from both functions.
    :param img_path: Image input
    :return:
    """
    print("Compare LK & Hierarchical LK - START")

    # Load and preprocess the images
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=0.5, fy=0.5)

    # Generate the transformed image
    transformation = np.array([[1, 0, -0.2],
                               [0, 1, -0.1],
                               [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, transformation, img_1.shape[::-1])

    # Compute optical flow using Lucas-Kanade method
    pts1, uv1 = opticalFlow(img_1.astype(np.float), img_2.astype(np.float), step_size=20, win_size=5)

    # Compute optical flow using Pyramidal Lucas-Kanade method
    arr3d = opticalFlowPyrLK(img_1.astype(np.float), img_2.astype(np.float), stepSize=20, winSize=5, k=5)
    pts2, uv2 = convert_3d_array_to_points(arr3d)

    # Display the optical flow results
    display_optical_flow_comparison(img_2, pts1, uv1, img_2, pts2, uv2)

    print("Compare LK & Hierarchical LK - END")


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()



# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------
def compare_translation_lk(image_path):
    """
    Compare the results of translation using Lucas-Kanade and Translation correlation methods.
    :param image_path: Input image path.
    :return:
    """
    print("compare_translation_lk - START")
    # Read and preprocess input image
    image1 = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite('imTransA1.jpg', cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Define translation matrix
    translation_matrix = np.array([[1, 0, -5],
                                   [0, 1, -5],
                                   [0, 0, 1]], dtype=np.float)

    # Apply translation to image1
    image2 = cv2.warpPerspective(image1, translation_matrix, image1.shape[::-1])

    # Find translation matrix using Lucas-Kanade
    lk_matrix = findTranslationLK(image1, image2)

    # Apply the obtained translation matrix to image1
    lk_image = cv2.warpPerspective(image1, lk_matrix, image1.shape[::-1])

    # Display results
    cv2.imshow("Translation from cv2", image2)
    cv2.imshow("Translation using Lucas-Kanade", lk_image)
    cv2.imwrite('imTransA2.jpg', cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("Translation matrix (Lucas-Kanade):\n", lk_matrix)
    cv2.waitKey(0)

    print("compare_translation_lk - END")

def translation_correlation(image_path):
    """
    Compare the translation results using Lucas-Kanade and translation correlation methods.
    :param image_path: Image input.
    :return:
    """
    print("Translation correlation - START")

    # Read and preprocess the input image
    image1 = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite('imTransB1.jpg', cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Define translation matrix
    translation_matrix = np.array([[1, 0, -20],
                                   [0, 1, -20],
                                   [0, 0, 1]], dtype=np.float)

    # Apply translation to image1
    image2 = cv2.warpPerspective(image1, translation_matrix, image1.shape[::-1])

    # Find translation matrix using correlation method
    correlation_matrix = findTranslationCorr(image1, image2)

    # Convert correlation matrix to the expected format
    correlation_matrix = np.vstack([correlation_matrix, [0, 0, 1]])

    # Apply the obtained translation matrix to image1
    correlated_image = cv2.warpPerspective(image1, correlation_matrix, image1.shape[::-1])

    # Display results
    cv2.imshow("Translation from cv2", image2)
    cv2.imshow("Translation using Correlation", correlated_image)
    cv2.imwrite('imTransB2.jpg', cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("Translation matrix (Correlation):\n", correlation_matrix)

    cv2.waitKey(0)

    print("Translation correlation - END")

def rigid_lk(image_path):
    """
    Compare the results of rigid transformation using Lucas-Kanade and RigidLK methods.
    :param image_path: Input image path.
    :return:
    """
    print("Rigid LK - START")

    # Read and preprocess the input image
    image1 = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite('imRigidA1.jpg', cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Define rigid transformation matrix
    theta = np.radians(3.5)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    translation_matrix = np.array([[cos_theta, -sin_theta, -0.5],
                                   [sin_theta, cos_theta, -0.5],
                                   [0.0, 0.0, 1.0]], dtype=np.float)

    # Apply rigid transformation to image1
    image2 = cv2.warpPerspective(image1, translation_matrix, image1.shape[::-1])

    # Find rigid transformation matrix using RigidLK
    rigid_matrix = findRigidLK(image1, image2)

    # Apply the obtained rigid transformation matrix to image1
    rigid_image = cv2.warpPerspective(image1, rigid_matrix, image1.shape[::-1])

    # Display results
    cv2.imshow("Rigid from cv2", image2)
    cv2.imshow("Rigid using RigidLK", rigid_image)
    cv2.imwrite('imRigidA2.jpg', cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("Rigid transformation matrix (RigidLK):\n", rigid_matrix)

    cv2.waitKey(0)

    print("Rigid LK - END")

def rigid_correlation(image_path):
    """
    Compare the results of rigid transformation using correlation and RigidCorr methods.
    :param image_path: Input image path.
    :return:
    """
    print("Rigid Correlation - START")

    # Read and preprocess the input image
    image1 = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, (0, 0), fx=2.5, fy=2.5)
    cv2.imwrite('imRigidB1.jpg', cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_GRAY2BGR))

    # Define rigid transformation matrix
    theta = np.radians(3.6)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    translation_matrix = np.array([[cos_theta, -sin_theta, 5],
                                   [sin_theta, cos_theta, 6],
                                   [0.0, 0.0, 1.0]], dtype=np.float)

    # Apply rigid transformation to image1
    image2 = cv2.warpPerspective(image1, translation_matrix, image1.shape[::-1])

    # Find rigid transformation matrix using RigidCorr
    rigid_matrix = findRigidCorr(image1, image2)

    # Apply the obtained rigid transformation matrix to image1
    rigid_image = cv2.warpPerspective(image1, rigid_matrix, image1.shape[::-1])

    # Display results
    cv2.imshow("Rigid from cv2", image2)
    cv2.imshow("Rigid using RigidCorr", rigid_image)
    cv2.imwrite('imRigidB2.jpg', cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    print("Rigid transformation matrix (RigidCorr):\n", rigid_matrix)

    cv2.waitKey(0)

    print("Rigid Correlation - END")


def imageWarpingDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Image Warping Demo - START")

    # Read and preprocess the input image
    input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    input_image = cv2.resize(input_image, (0, 0), fx=0.5, fy=0.5)

    # Define the transformation matrix
    transformation_matrix = np.array([[0.9, 0, 0],
                                      [0, 0.9, 0],
                                      [0, 0, 1]], dtype=np.float)

    # Apply the predefined transformation matrix to the input image using cv2.warpPerspective
    warped_image_cv2 = cv2.warpPerspective(input_image, transformation_matrix, input_image.shape[::-1])

    # Apply the predefined transformation matrix to the input image using a custom function
    warped_image_custom = warpImages(input_image, warped_image_cv2, transformation_matrix)

    # Display the original photo, warped image from cv2, and the custom inverse warp
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(input_image, cmap='gray')
    ax[0].set_title('Original Photo')
    ax[1].imshow(warped_image_cv2, cmap='gray')
    ax[1].set_title('Warped Image (cv2)')
    ax[2].imshow(warped_image_custom, cmap='gray')
    ax[2].set_title('Custom Inverse Warp')
    plt.show()

    cv2.waitKey(0)

    print("Image Warping Demo - END")


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("Gaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("Laplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():
    im1 = cv2.cvtColor(cv2.imread('input/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

# ---------------------------------------------------------------------------
# --------------------- My Helper Functions ---------------------
# ---------------------------------------------------------------------------

def convert_3d_array_to_points(mat: np.ndarray) -> (np.ndarray, np.ndarray):
    x_y_return = []
    u_x_return = []
    for x in range(len(mat)):
        for y in range(len(mat[0])):
            dummy = mat[x, y]
            if dummy[0] != 0 or dummy[1] != 0:
                x_y_return.append([y, x])
                u_x_return.append(dummy)
    return np.array(x_y_return), np.array(u_x_return)


def display_optical_flow_comparison(img1: np.ndarray, pts1: np.ndarray, uvs1: np.ndarray,
                                    img2: np.ndarray, pts2: np.ndarray, uvs2: np.ndarray):
    """
    Display a comparison of optical flow results for two images.
    :param img1: First input image
    :param pts1: Points in the first image
    :param uvs1: Optical flow vectors in the first image
    :param img2: Second input image
    :param pts2: Points in the second image
    :param uvs2: Optical flow vectors in the second image
    :return: None
    """
    fig, axes = plt.subplots(1, 2)

    # Display optical flow for first image
    axes[0].imshow(img1, cmap='gray')
    axes[0].quiver(pts1[:, 0], pts1[:, 1], uvs1[:, 0], uvs1[:, 1], color='r')
    axes[0].set_title('Optical Flow (Lucas-Kanade)')

    # Display optical flow for second image
    axes[1].imshow(img2, cmap='gray')
    axes[1].quiver(pts2[:, 0], pts2[:, 1], uvs2[:, 0], uvs2[:, 1], color='r')
    axes[1].set_title('Optical Flow (Hierarchical Lucas-Kanade)')

    plt.show()


def main():
    print("ID:", myID())

    img_path = 'input/boxMan.jpg'
    lkDemo(img_path)
    hierarchicalkDemo(img_path)
    compareLK(img_path)
    #
    # # Translation Comparison
    compare_translation_lk('input/japan.jpg')
    #
    # # Translation Correlation
    translation_correlation('input/shrekArt.jpg')
    #
    # # Rigid
    rigid_lk('input/maldives.jpg')
    rigid_correlation('input/tigerArt.jpg')
    #
    # # Image Warping
    imageWarpingDemo(img_path)

    pyrGaussianDemo('input/pyr_bit.jpg')
    pyrLaplacianDemo('input/pyr_bit.jpg')
    blendDemo()


if __name__ == '__main__':
    main()
