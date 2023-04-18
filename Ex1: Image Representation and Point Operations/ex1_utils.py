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


def myID() -> int:
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
    image = imReadAndConvert(filename, representation)
    plt.imshow(image)
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
    global imgYIQ
    image = np.copy(imgOrig)  # making a copy of original the image
    image_shape = image.shape  # get the shape of the image
    isRGB = False  # init a flag to check if the image is RGB or not

    # Check if the image has more than 2 dimensions. If so, it is RGB:
    if len(image_shape) > 2:
        # Convert image from RGB to YIQ
        imgYIQ = transformRGB2YIQ(image)
        # extracting the Y channel from imgYIQ (greyscale version of the image)
        image = imgYIQ[..., 0]
        isRGB = True

    # Normalize the pixel values between 0 and 255
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    # Round the normalized pixel values to the nearest integer
    image = (np.around(image)).astype('uint8')

    # Calculating the image's histogram (range=[0,255]):
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
    look_up_table = (np.floor(cumulative_sum_normalized * 255)).astype('uint8')

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
    image_equalized = (np.around(image_equalized)).astype('uint8')
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
        imgYIQ[..., 0] = image_equalized / 255
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
    # Check if image is RGB and convert to grayscale if necessary:
    if imOrig.ndim == 3 and imOrig.shape[-1] == 3:
        isRGB = True
        imgYIQ = transformRGB2YIQ(imOrig)
        image = imgYIQ[..., 0]
    else:
        image = np.copy(imOrig)
        isRGB = False

    # Initializing he lists to return:
    images = []
    errors = []

    # Normalizing the pixels of image from [0,1] to [0,255]
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = image.astype('uint8')
    # Calculating the histogram of the image
    histogram, bins = np.histogram(image, 256, [0, 255])

    # Creating borders for quantization process:
    borders = np.zeros(nQuant + 1, dtype=int)
    for i in range(nQuant + 1):
        borders[i] = i * (255.0 / nQuant)
    borders[-1] = 256

    # Main loop:
    for i in range(nIter):
        # Create array to store the value of histogram counts for each border
        quantization_levels = np.zeros(nQuant, dtype=int)

        # Calculate the value of histogram counts of pixel intensities within the border:
        for j in range(nQuant):
            # create an array containing the histogram counts for each pixel intensity
            hist_counts = histogram[borders[j]:borders[j + 1]]
            # create an array containing the pixel intensities
            pixel_intensity_range = np.arange(int(borders[j]), int(borders[j + 1]))
            quantization_levels[j] = (pixel_intensity_range * hist_counts).sum() / (hist_counts.sum().astype(int))

        # Recalculate the borders of the partitions based on the average of the quantization levels"
        # Store the first and last elements of the current border array
        z_first = borders[0]
        z_last = borders[-1]
        # Initialize a new array for the updated borders
        borders = np.zeros_like(borders)
        # Loop through each partition  (except first and last) and calculate new border value:
        for k in range(1, nQuant):
            borders[k] = (quantization_levels[k - 1] + quantization_levels[k]) / 2

        # Update the first and last values
        borders[0] = z_first
        borders[-1] = z_last

        # Recoloring the image:
        # create new array with the same shape and sata type as image
        temporary_image = np.zeros_like(image)
        for l in range(nQuant):
            quant_level_boundary = borders[l]
            # recolor the pixels of the temp image based on their intensity values in the original image
            temporary_image[image > quant_level_boundary] = quantization_levels[l]

        images.append(temporary_image)

        # Calculate the Mean Squared Error between rhe original image and the quantized image:
        errors.append(np.sqrt((image - temporary_image) ** 2).mean())
        # if the absolute difference between the last two elements in the errors array is less than 0.001, them converged.
        if len(errors) > 1 and abs(errors[-2] - errors[-1]) < 0.001:
            break

    # Check if input image is RGB. If so, convert it from YIQ to RGB before added to the imaged list.
    if isRGB:
        for i in range(len(images)):
            # normalize the luminance channel (Y) of YIQ image
            imgYIQ[..., 0] = images[i] / 255
            # convert the YIQ image back to RGB
            images[i] = transformYIQ2RGB(imgYIQ)
            # clip the RGB values of the current image to the range of [0,1].
            images[i][images[i] > 1] = 1
            images[i][images[i] < 0] = 0

    return images, errors

