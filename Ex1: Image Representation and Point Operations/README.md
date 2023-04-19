# Ex1: Image Representation and Point Operations


### Table of Content
* [About The Project](About-The-Project)
* [Software Information](Software-Information)
* [Submission Files](Submission-Files)
* [Functions](Functions)
* [Screenshots](Screenshots)




### About The Project
The main purpose of the exercise was to get us acquainted with some of Python's image processing facilities. This exersice covers:
- Loading grayscale and RGB image representations.
- Displaying figures and images.
- Transforming RGB color images back and forth from the YIQ color space.
- Performing intensity transformations: histogram equalization.
- Performing optimal equalization.

---

### Software Information

Python Version - Python 3.8.<br>
Open-CV Version - 4.5.5<br>
Matplotlib Version - 3.7.1<br>
IDE - PyCharm Community

---

### Submission Files
* **ex1_main.py** - The driver code. This code was given by the professor. Inside the file lays the main script for running the code.
It is also checks that all the wanted functions were implemented as required. 
  
* **ex1_utils.py** - The primary implementation class.

* **gamma.py** - This file contains a function that performs gamma correction in an image with a given gamma.

* **swiss_boat.jpg** - A photo I took while traveling Switzerland, for testing purposes.

* **README.md** - My readme file.

---

### Functions

- **myID() -> int:**

    Returns my ID for checking purposes.


- **imReadAndConvert(filename: str, representation: int) -> np.ndarray:**

    Reads an image, and returns in converted as requested


- **imDisplay(filename: str, representation: int):**

    Reads an image as RGB or GRAY_SCALE and displays it
    

- **transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:**

    Converts an RGB image to YIQ color space


- **transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:**
      
    Converts an YIQ image to RGB color space

  
- **histogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):**
     
    Equalizes the histogram of an image and creates a new image fixed according to the histogram Equalization

    
- **quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):**
      
    Converts an Image to a quantize color scheme. Returns A list of errors which we made each iteration and a list of images.


- **gammaDisplay(img_path: str, rep: int) -> None:**

    GUI for gamma correction


- **trackBar(brightness: int):**

   Applies gamma correction to an input image based on the user-specified brightness value, and displays the corrected image in a window using OpenCV.


---

### Screenshots

* Basic images display in RGB and YIQ.

![image](https://user-images.githubusercontent.com/77110578/233001467-f224a8bb-0c60-4bda-8fdc-fbfd91d84bf5.png)

---
* RGB to YIQ conversion

![image](https://user-images.githubusercontent.com/77110578/233000563-13d77a10-b2bb-4f9f-aff2-68c229e41897.png)

--- 
* Histogram Equalization for YIQ image

![image](https://user-images.githubusercontent.com/77110578/233001013-cb5b4113-7019-443e-8e44-a98f719dd439.png)

---
* Histogram Equalization for RGB image


![image](https://user-images.githubusercontent.com/77110578/233001837-652d8c39-f34f-48b9-83cd-5e23d74a4193.png)

---
* Optimal Image Quantization for YIQ image

![image](https://user-images.githubusercontent.com/77110578/233007107-89b302c5-622f-4da1-afe8-651f9c474f01.png)

---
* Optimal Image Quantization for RGB image

![image](https://user-images.githubusercontent.com/77110578/233007405-0ac7f7c7-6f99-4f70-ba0e-7db568314524.png)

---
* Gamma Display
  
  In Gamma Correction, it the gamma value is higher, the image will be brighter in a grayscale picture. The reason for that is becasue the higher gamma value applies a stronger correction to the pixels intensities.

![image](https://user-images.githubusercontent.com/77110578/233003477-f3dd15ae-befe-416e-b4ef-05f4bd7444bd.png)