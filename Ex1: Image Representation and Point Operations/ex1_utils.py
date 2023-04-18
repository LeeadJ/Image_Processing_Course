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
from typing import List
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
YIQ_MATRIX = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 313308785


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """

    image = cv2.imread(filename)
    if representation == LOAD_GRAY_SCALE:  # (1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif representation == LOAD_RGB:  # (2)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    image = imReadAndConvert((filename, representation))
    plt.show(image)
    if representation == LOAD_GRAY_SCALE:
        plt.gray()
    plt.show()



def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    return np.dot(imgRGB, YIQ_MATRIX.transpose())


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    return np.dot(imgYIQ, np.linalg.inv(YIQ_MATRIX).transpose())


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # check if image is RGB.
    image = np.copy(imgOrig) # making a copy of original the image
    image_shape = image.shape # get the shape of the image
    isRGB = False # init a flag to check if the image is RGB or not

    # Check if the image has more than 2 dimensions. If so, it is RGB:
    if len(image_shape) > 2:
        # Convert image from RGB to YIQ
        imgYIQ = transformRGB2YIQ(image)
        # extracting the Y channel from imgYIQ (greyscale version of the image)
        image = imgYIQ[:, :, 0]
        isRGB = True

    # Normalize the pixel values between 0 and 255
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    # Round the normalized pixel values to the nearest integer
    image = (np.around(image)).astype('unit8')

    # Calculating the image's histogram:
    # Flatten the original image to a 1d array
    image_flattened = image.ravel()
    # Create an empty histogram array with 256 bins
    histogram = np.zeros(256)
    # Loop through each pixel value in the flattened image and update the histogram
    for pixel_value in image_flattened:
        histogram[pixel_value] += 1

    # Calculate the cumulative sum of the histogram:
    # Create an array of zeros with the same shape as the histogram
    cumulative_sum = np.zeros_like(histogram)
    # Set the first element of the cumulative sum array to be the same as the first element of the histogram
    cumulative_sum[0] = histogram[0]
    # Loop through the remaining elements of the histogram and calculate the cumulative sum
    for i in range(1, len(histogram)):
        cumulative_sum[i] = histogram[i] + cumulative_sum[i - 1]

    # Create a lookup table for intensity transformation:
    # Normalize cumulative sum
    cumulative_sum_normalized = cumulative_sum / cumulative_sum.max()
    # create a lookup table using normalized cumulative sum
    look_up_table = (np.floor(cumulative_sum_normalized * 255)).astype('unit8')

    # Apply the intensity transformation to the input image:
    # Create an empty array for the equalized image
    image_equalized = np.zeros_like(image, dtype=float)
    # replace each intensity i with its corresponding intensity value in the lookup table
    for i in range(256):
        image_equalized[image == i] = look_up_table[i]

    # Normalize and calculate the histogram of the equalized image
    # normalize the equalized image
    image_equalized = cv2.normalize(image_equalized, None, 0, 255, cv2.NORM_MINMAX)
    # convert the equalized image to 8-bit integer type
    image_equalized = (np.around(image_equalized)).astype('unit8')
    # flatten the equalized image
    image_equalized_flat = image_equalized.ravel()
    # create an empty array for the histogram of the equalized image
    histogram_equalized = np.zeros(256)
    # calculate the histogram of the equalized image
    for i in image_equalized_flat:
        histogram_equalized[i] += 1

    # If the original image was RGB, convert it back to RGB after equalization
    if isRGB:
        # Convert Y channel back to [0,1] range and replace it in the YIQ array
        imgYIQ[:, :, 0] = image_equalized / 255
        # Convert the YIQ image back to RGB and return it
        image_equalized = transformYIQ2RGB(imgYIQ)

    return image_equalized, histogram, histogram_equalized


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
