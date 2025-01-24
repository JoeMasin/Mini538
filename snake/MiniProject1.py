#import pandas
#import scipy 
#import matplotlib
#import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import filedialog

def load_input():
    """
    Loads an input image or video to be processed & displays the image or video
    :return: filename: file name of input image , data: Image or movie 
    """
    
    # initialize tkinter root window
    root = tk.Tk()
    root.withdraw()
    
    # get input image file from the user
    input_file = filedialog.askopenfilename(initialdir="./",
                                          title="Select an Input Image")
    try:
        # read & display the image in grayscale
        img_grayscale = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
        if img_grayscale is None:
            raise Exception("ERROR: CV Unable to load image file")
        #cv.imshow("Grayscale Input Image", img_grayscale)
        #cv.waitKey(0)
        #cv.imwrite("Grayscale Input Image.png", img_grayscale)

    except Exception as e:
        # if OpenCV image input can't be read, display an error message 
        print(e)
        try:
            #Create a videoCapture object that should contain all Frames 
            video = cv.VideoCapture(input_file)

            # Check if the video opened successfully
            if not video.isOpened():
                raise Exception("")
            
            frames = []
            # Read through the video frame by frame and save the individual frames 
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    raise Exception("ERROR: Frame not read correctly")

                gray_frame= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # Process the frame here (e.g., display it)

                cv.imshow('Frame', gray_frame)
                frames.append(gray_frame)

                # Press 'q' to exit
                if cv.waitKey(1) == ord('q'):
                    break

            # Release the VideoCapture object
            video.release()
            cv.destroyAllWindows()

        except Exception as e:
            print(e)
        print("# of Frames: ", len(frames))    
        return input_file, frames

    return input_file, img_grayscale

def IdentMat(image):
    IR = image.shape(0)
    print(IR)
    I = np.linalg.inv(IR)
    
    return(I)

def main():
    print("Will you be inoputing a image or video?:")
    input()
    test1, test2 = load_input()
    print(test2)
            


main()