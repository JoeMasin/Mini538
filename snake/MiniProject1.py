import pandas
import scipy 
import matplotlib
import math
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import filedialog

def load_image():
    """
    Loads an input image to be processed & displays the image
    :return: filename: file name of input image
    """

    # initialize tkinter root window
    root = tk.Tk()
    root.withdraw()

    # get input image file from the user
    input_file = filedialog.askopenfilename(initialdir=".",
                                          title="Select an Input Image")

    try:
        # read & display the image in grayscale
        img_grayscale = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
        # cv.imshow("Grayscale Input Image", img_grayscale)
        cv.imwrite("Grayscale Input Image.png", img_grayscale)

    except ValueError:
        # if image can't be read, display an error message and exit the program
        print("ERROR: Unable to load image file")
        exit(0)

    return input_file, img_grayscale

def IdentMat(image):
    IR = image.shape(0)
    print(IR)
    I = numpy.linalg.inv(IR)
    
    return(I)

def main():
    image, gray = load_image
    
    cv.show(image)

main()