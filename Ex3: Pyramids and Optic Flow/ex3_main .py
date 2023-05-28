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
    print("Compare LK & Hierarchical LK")

    pass


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def imageWarpingDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Image Warping Demo")

    pass


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

def main():
    print("ID:", myID())

    img_path = 'input/boxMan.jpg'
    lkDemo(img_path)
    hierarchicalkDemo(img_path)
    # compareLK(img_path)
    #
    # imageWarpingDemo(img_path)
    #
    # pyrGaussianDemo('input/pyr_bit.jpg')
    # pyrLaplacianDemo('input/pyr_bit.jpg')
    # blendDemo()


if __name__ == '__main__':
    main()
