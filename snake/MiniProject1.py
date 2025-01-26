#import pandas
#import scipy 
#import matplotlib
#import math
from unittest import result
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import filedialog

def load_input(): # ME
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
            img_grayscale = np.loadtxt(input_file)
            
            if img_grayscale is None:
                raise Exception("ERROR: numpy not able to load image file")
        
        except Exception as e:
            print(e)

            try:
                #Create a videoCapture object that should contain all Frames 
                video = cv.VideoCapture(input_file)

                # Check if the video opened successfully
                if not video.isOpened():
                    raise Exception("ERROR: Video not opened correctly")
                
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

def IdentMat(image): # ME
    rows = image.shape[0]
    columns = image.shape[1]
    Isize = max(rows, columns)
    I = np.identity(Isize)
    return(I)

def ForwardEulerImage(image):#, time_step, alpha, beta): ME 
    #Space related Forward Euler Method using 2 spacial steps forward and back 
    rows = image.shape[0]
    columns = image.shape[1]
    I_Mat = IdentMat(image)
    
    alpha = 1
    A = [0, 1, -2, 1, 0]
    
    beta = 0.5
    B = [-1, 4, -6, 4, -1]

    delta_T = 1

    combined = alpha@A + beta@B
    print(combined)

    combined1 = combined@delta_T

    #for i in range(max(rows,columns)):
        #print(i)
        
            #tobemult = image[i-2]
    new_image = np.dot(combined1, )
        #new_image = (I_Mat + time_step*)
    #np.linalg.inv(result)
    return(image)

def internal_force(snake, alpha, beta): # AI
    """
    Compute the internal force of a snake (active contour).
    
    Parameters:
    snake: ndarray of shape (N, 2)
        Coordinates of the snake's points [(x1, y1), (x2, y2), ...].
    alpha: float
        Weight for the elasticity term.
    beta: float
        Weight for the bending term.
    
    Returns:
    F_internal: ndarray of shape (N, 2)
        Internal forces acting on each point of the snake.
    """
    N = len(snake)
    F_internal = np.zeros_like(snake)
    
    for i in range(N):
        # Get neighbors with periodic boundary conditions
        v_prev = snake[i - 1]           # Previous point
        v_curr = snake[i]               # Current point
        v_next = snake[(i + 1) % N]     # Next point
        v_prev2 = snake[i - 2]          # Second previous point
        v_next2 = snake[(i + 2) % N]    # Second next point
        
        # Elasticity force
        elasticity = alpha * (v_prev - 2 * v_curr + v_next)
        
        # Bending force
        bending = beta * (v_prev2 - 4 * v_prev + 6 * v_curr - 4 * v_next + v_next2)
        
        # Total internal force
        F_internal[i] = elasticity + bending
    
    return F_internal


def internal_energy(snake, alpha, beta): # AI
    """
    Compute the internal energy of a snake (active contour).
    
    Parameters:
    snake: ndarray of shape (N, 2)
        Coordinates of the snake's points [(x1, y1), (x2, y2), ...].
    alpha: float
        Weight for the elasticity term.
    beta: float
        Weight for the bending term.
    
    Returns:
    E_internal: float
        Total internal energy of the snake.
    """
    N = len(snake)
    E_internal = 0.0
    
    for i in range(N):
        # Get neighbors with periodic boundary conditions
        v_prev = snake[i - 1]
        v_curr = snake[i]
        v_next = snake[(i + 1) % N]
        
        # Elasticity term (first derivative)
        elasticity = np.linalg.norm(v_next - v_curr)**2
        
        # Bending term (second derivative)
        bending = np.linalg.norm(v_next - 2 * v_curr + v_prev)**2
        
        # Add to total energy
        E_internal += alpha * elasticity + beta * bending
    
    return E_internal


def backward_euler(snake, alpha, beta, delta_t): # AI
    """
    Update the snake using the Backward Euler method for internal forces.
    
    Parameters:
    snake: ndarray of shape (N, 2)
        Coordinates of the snake's points [(x1, y1), (x2, y2), ...].
    alpha: float
        Weight for the elasticity term.
    beta: float
        Weight for the bending term.
    delta_t: float
        Time step.
    
    Returns:
    snake_updated: ndarray of shape (N, 2)
        Updated coordinates of the snake's points.
    """
    N = len(snake)
    I = np.eye(N)  # Identity matrix
    
    # First-order finite difference matrix (elasticity)
    D1 = np.roll(I, -1, axis=0) - 2 * I + np.roll(I, 1, axis=0)
    
    # Second-order finite difference matrix (bending)
    D2 = np.roll(I, -2, axis=0) - 4 * np.roll(I, -1, axis=0) + 6 * I - 4 * np.roll(I, 1, axis=0) + np.roll(I, 2, axis=0)
    
    # Internal force matrix
    A = alpha * D1 + beta * D2
    
    # Backward Euler matrix
    BE_matrix = I - delta_t * A
    
    # Solve for v^{n+1}
    snake_updated = np.linalg.solve(BE_matrix, snake)
    
    return snake_updated

def forward_euler(snake, alpha, beta, delta_t): # AI
    """
    Update the snake using the Forward Euler method for internal forces.
    
    Parameters:
    snake: ndarray of shape (N, 2)
        Coordinates of the snake's points [(x1, y1), (x2, y2), ...].
    alpha: float
        Weight for the elasticity term.
    beta: float
        Weight for the bending term.
    delta_t: float
        Time step.
    
    Returns:
    snake_updated: ndarray of shape (N, 2)
        Updated coordinates of the snake's points.
    """
    F_internal = internal_force(snake, alpha, beta)
    # Update positions using Forward Euler
    snake_updated = snake + delta_t * F_internal

    return snake_updated

def main(): # ME
    #print("Will you be inoputing a image (0) or video(1)?:")
    #type = input()
    #if type == "1":
    Input_FileName, Inital_data = load_input()
    plt.plot(Inital_data[:,0], Inital_data[:,1])
    plt.show()
    
    alpha = 1.0  # Elasticity weight
    beta = 0.5   # Bending weight
    energy = internal_energy(Inital_data, alpha, beta)
    print(energy)
    F_internal = internal_force(Inital_data, alpha, beta)
    # Plot internal forces as arrows
    for i in range(len(Inital_data)):
        plt.quiver(Inital_data[i, 0], Inital_data[i, 1], 
                F_internal[i, 0], F_internal[i, 1], 
                angles='xy', scale_units='xy', scale=1, color='r')

    plt.title("Snake and Internal Forces")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()
    
    alpha = 1.0   # Elasticity weight
    beta = 0.5    # Bending weight
    delta_t = 0.01 # Time step

    # Update snake using Backward Euler
    snake_updated = backward_euler(Inital_data, alpha, beta, delta_t)
    for i in range(0, 5):
        snake_updated = backward_euler(snake_updated, alpha, beta, delta_t)
        # Plot the original and updated snake
        plt.figure()
        plt.plot(Inital_data[:, 0], Inital_data[:, 1], 'o-', label="Original Snake")
        plt.plot(snake_updated[:, 0], snake_updated[:, 1], 'o-', label="Updated Snake")
        plt.title("Snake Evolution (Backward Euler)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.axis('equal')
        plt.show()

        # Plot the original and updated snake

    snake_updated = forward_euler(Inital_data, alpha, beta, delta_t)
    plt.figure()
    plt.plot(Inital_data[:, 0], Inital_data[:, 1], 'o-', label="Original Snake")
    plt.plot(snake_updated[:, 0], snake_updated[:, 1], 'o-', label="Updated Snake")
    plt.title("Snake Evolution (Forward Euler)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

main()