"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from ex1_utils import LOAD_GRAY_SCALE
import cv2 as cv2
import numpy as np

title_window = 'Gamma correction'
max_gamma_slider = 200
MAX_PIXEL = 255


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global image
    if rep == LOAD_GRAY_SCALE:
        image = cv2.imread(img_path, 2)
    else:
        image = cv2.imread(img_path, 1)

    cv2.namedWindow(title_window)
    trackbar_name = 'Gamma %d' % max_gamma_slider
    cv2.createTrackbar(trackbar_name, title_window, 100, max_gamma_slider, trackBar)
    trackBar(100)
    cv2.waitKey()


def trackBar(brightness: int):
    gamma = float(brightness) / 100
    invert_gamma = 1000 if gamma == 0 else 1.0 / gamma
    gammaTable = np.array([((i / float(MAX_PIXEL)) ** invert_gamma) * MAX_PIXEL
                           for i in np.arange(0, MAX_PIXEL + 1)]).astype('uint8')
    img = cv2.LUT(image, gammaTable)
    cv2.imshow(title_window, img)


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
