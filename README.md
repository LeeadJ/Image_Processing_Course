# Image Processig Course

This README provides a summary of three different README files related to image processing and computer vision exercises.

## Exercise 1: Image Representation and Point Operations

### About The Project
The purpose of this exercise is to familiarize oneself with Python's image processing capabilities. The exercise covers various tasks, including loading grayscale and RGB images, displaying figures and images, transforming RGB images to the YIQ color space and vice versa, performing intensity transformations like histogram equalization, and optimal equalization.

### Software Information
- Python Version: 3.8
- OpenCV Version: 4.5.5
- Matplotlib Version: 3.7.1
- IDE: PyCharm Community

### Submission Files
- ex1_main.py: The driver code provided by the professor, containing the main script.
- ex1_utils.py: The primary implementation class.
- gamma.py: Contains a function for performing gamma correction on an image.
- testImg1.jpg: A test image.
- testImg2.jpg: Another test image.
- README.md: The readme file.

### Functions
The exercise provides several functions, including `myID()`, `imReadAndConvert()`, `imDisplay()`, `transformRGB2YIQ()`, `transformYIQ2RGB()`, `histogramEqualize()`, `quantizeImage()`, `gammaDisplay()`, and `trackBar()`. Each function serves a specific purpose related to image processing and point operations.

### Screenshots
The README includes screenshots showcasing the basic display of images in RGB and YIQ, RGB to YIQ conversion, histogram equalization for YIQ and RGB images, optimal image quantization for YIQ and RGB images, and gamma display.
![image](https://user-images.githubusercontent.com/77110578/233001467-f224a8bb-0c60-4bda-8fdc-fbfd91d84bf5.png)
![image](https://user-images.githubusercontent.com/77110578/233001837-652d8c39-f34f-48b9-83cd-5e23d74a4193.png)
![image](https://user-images.githubusercontent.com/77110578/233003477-f3dd15ae-befe-416e-b4ef-05f4bd7444bd.png)

## Exercise 2: Convolution & Edge Detection

### About The Project
The purpose of this exercise is to understand the concept of convolution and edge detection by performing operations on images. The exercise covers topics such as implementing convolution for 1D and 2D arrays, image derivatives, image blurring, edge detection using different methods, Hough circles transformation, and bilateral filtering.

### Contents
The assignment includes several parts:
- Convolution: Implementation of convolution for 1D and 2D signals.
- Image derivatives & blurring: Calculation of image gradient magnitude and direction, and implementation of image blurring using convolution with a Gaussian kernel.
- Edge detection: Implementation of edge detection algorithms.
- Hough Circles: Implementation of the Hough circles transform.
- Bilateral filter: Implementation of the Bilateral filter.

### Requirements
The implementation requires the following dependencies:
- numpy
- OpenCV

### Usage
To run the code, the required dependencies need to be installed. Then, the repository should be cloned, and the desired exercise file can be executed.

### Screenshots
The README includes screenshots showcasing the results of different operations, such as convolution, image derivatives, blurring, edge detection, Hough circles detection, and bilateral filtering.


## Exercise 3: Pyramids and Optic Flow

### About The Project
This project focuses on implementing tasks related to pyramids and optic flow. It provides functions for creating Gaussian and Laplacian pyramids, estimating optical flow using the Lucas-Kanade algorithm, image alignment and warping, and blending images using pyramid blending. The project aims to explore image processing and computer vision concepts, allowing for the analysis, manipulation, and estimation of motion in images.

### Contents
The project includes exercises related to:
- Lucas Kanade Optical Flow: Implementing the Lucas-Kanade algorithm for optical flow estimation.
- Hierarchical Lucas Kanade Optical Flow: Capturing large movements using a hierarchical approach to optical flow estimation.
- Image Alignment and Warping: Finding parameters for aligning and warping images.


