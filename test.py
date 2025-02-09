from statistics import NormalDist
from turtle import update
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from types import new_class
from unittest import result
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import Image, filedialog
import time
from PIL import Image
import skimage

def load_input(): # 
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
                # Read through the video frame by frame and save the individual frames into a variable
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
            return input_file, frames, False

    return input_file, img_grayscale, True

def compute_normals(contour):
    """
    Compute the normal vectors for a 2D contour.
    :param contour: A numpy array of shape (N, 2) representing the (x, y) coordinates of the contour.
    :return: A numpy array of shape (N, 2) representing the normal vectors at each point.
    """
    # Compute tangent vectors using central differences
    #tangent = np.roll(contour, -1, axis=0) - np.roll(contour, 1, axis=0)
    #
    ## Compute normal vectors by rotating tangent vectors by 90 degrees
    #normal = np.zeros_like(tangent)
    #normal[:, 0] = tangent[:, 1]  # N_x = -T_y
    #normal[:, 1] = -tangent[:, 0]   # N_y = T_x
    #
    ## Normalize the normal vectors
    #magnitude = np.linalg.norm(normal, axis=1, keepdims=True)
    #magnitude[magnitude == 0] = 1  # Avoid division by zero
    #unit_normal = normal / magnitude
    
        # calculate directions for each point in contour
    tangents = np.roll(contour, -1, axis=0) - np.roll(contour, 1, axis=0)

    # convert directions into unit normal vectors
    normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    N = normals / norms
    
    return N

def compute_mean_intensities(image, contour):
    """
    Compute the mean intensity inside and outside the contour.
    :param image: A 2D numpy array representing the image.
    :param contour: A numpy array of shape (N, 2) representing the (x, y) coordinates of the contour.
    :return: A tuple (mean_inside, mean_outside) representing the mean intensities.
    """
    # Create a mask for the region inside the contour
    rr, cc = polygon(contour[:, 0], contour[:, 1], image.shape)
    mask = np.zeros_like(image, dtype=bool)
    mask[rr, cc] = True
    
    # Compute mean intensities
    mean_inside = np.mean(image[mask])
    mean_outside = np.mean(image[~mask])
    mean = np.mean(image)
    
    return mean_inside, mean_outside, mean

def visualize_contour_with_normals(contour, normals, scale=1.0):
    """
    Visualize the contour and its normals using Matplotlib.
    :param contour: A numpy array of shape (N, 2) representing the (x, y) coordinates of the contour.
    :param normals: A numpy array of shape (N, 2) representing the normal vectors at each point.
    :param scale: Scaling factor for the normal vectors (to make them longer or shorter).
    """
    plt.figure(figsize=(8, 8))
    
    # Plot the contour
    plt.plot(contour[:, 0], contour[:, 1], 'b-', label='Contour')
    plt.scatter(contour[:, 0], contour[:, 1], c='red', label='Points')
    
    # Plot the normals using quiver
    plt.quiver(
        contour[:, 0],  # x-coordinates of the points
        contour[:, 1],  # y-coordinates of the points
        normals[:, 0],  # x-components of the normals
        normals[:, 1],  # y-components of the normals
        angles='xy', scale_units='xy', scale=scale, color='green', label='Normals'
    )
    
    # Set plot limits and labels
    plt.xlim(np.min(contour[:, 0]) - 1, np.max(contour[:, 0]) + 1)
    plt.ylim(np.min(contour[:, 1]) - 1, np.max(contour[:, 1]) + 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Contour with Normals')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_circle_snake(image, radius, n_points=100):
    # input image, radius of the snake ceneterd in the middle of the image, # of points in the snake

    height, width = image.shape[:2]
    
    # Find the center of the image
    center_x = width // 2
    center_y = height // 2
    
    # Generate the points on the circle
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    circle_points = np.array([(center_x + radius * np.cos(angle), center_y + radius * np.sin(angle)) 
                              for angle in angles])
    
    return circle_points

def get_pixel_values_at_snake_points(image, snake_points):

    """
    Extracts the pixel values from the image at the locations specified by the snake points.
    
    Parameters:
    - image: np.ndarray, input image (height x width, or height x width x 3 for color).
    - snake_points: np.ndarray, snake points (Nx2 array, where N is the number of points).
    
    Returns:
    - pixel_values: np.ndarray, extracted pixel values at the snake points.
    """

    # Round the coordinates to the nearest integer
    rounded_coordinates = np.round(snake_points).astype(int)

    # Extract pixel values at the rounded coordinates
    pixel_values = image[rounded_coordinates[:, 1], rounded_coordinates[:, 0]]
    return pixel_values 

def update_normals_Fext(snake , image): #  Fixed to work correctly 
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
    normals = compute_normals(snake)
    values_at_snake = get_pixel_values_at_snake_points(image, snake)

    mean_inside, mean_outside, mean = compute_mean_intensities(image, snake)
    mean_inside = mean_inside
    mean_outside = mean_outside
    mean = mean
    #D2 = np.roll(test, -1)  + test +  np.roll(test, 1) 
    values_at_snake = values_at_snake
    # Internal force matrix
    #A = D2 / 3
    Fext = 2 *(mean_inside - mean_outside) * ((values_at_snake) - ((1/2)*(mean_inside + mean_outside)))
    #Fext_out = np.diagonal(Fext)
    
    #for i in range(len(normals)):
    #    if test[i] > mean:
    #        normals_updated[i] = -1 * normals[i]
    #    else:
    #        normals_updated[i] = normals[i]
    #Fext = Fext * 2
    return Fext, normals

def backward_euler_Bint(snake, alpha, beta, delta_t): #  Fixed to work correctly 
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
    D1 = -1 * np.roll(I, -1, axis=0) + 2 * I - 1 * np.roll(I, 1, axis=0)
    
    # Second-order finite difference matrix (bending)
    D2 = -1 * np.roll(I, -2, axis=0) + 4 * np.roll(I, -1, axis=0) - 6 * I + 4 * np.roll(I, 1, axis=0)  - 1* np.roll(I, 2, axis=0)
    
    # Internal force matrix
    A = alpha * D1 + beta * D2
    
    # Backward Euler matrix
    BE_matrix = np.linalg.inv(I - delta_t * A)
    return BE_matrix

def resample_contour(contour, spacing=5):
    # Ensure contour is closed by appending the first point
    closed_contour = np.vstack([contour, contour[0]])

    # Calculate cumulative distance (include wrap-around)
    diff = np.diff(closed_contour, axis=0)
    dist = np.linalg.norm(diff, axis=1)
    cum_dist = np.insert(np.cumsum(dist), 0, 0)
    total_dist = cum_dist[-1]
    
    # Generate new points with explicit wrap-around
    num_points = int(np.round(total_dist / spacing))
    new_distances = np.linspace(0, total_dist, num_points, endpoint=False)
    
    new_contour = np.zeros((num_points, 2))
    for i, d in enumerate(new_distances):
        idx = np.searchsorted(cum_dist, d, side='right') - 1
        t = (d - cum_dist[idx]) / (cum_dist[idx+1] - cum_dist[idx])
        new_contour[i] = closed_contour[idx] + t * (closed_contour[idx+1] - closed_contour[idx])
    
    return new_contour

def enforce_bounds(points, img_shape):
    points[:, 0] = np.clip(points[:, 0], 0, img_shape[1]-1)
    points[:, 1] = np.clip(points[:, 1], 0, img_shape[0]-1)
    return points

def Active_Contour(alpha, beta, delta_t, newsnake, image, itterations, pause_time):
    
    for i in range(0, itterations):
        #I = np.eye(N)
        #Fext = (image - (1/2)*((mean_inside) + (mean_outside)))
        #adjust_normals = np.diagonal(Fext)
        Bint = backward_euler_Bint(newsnake, alpha, beta, delta_t)

        #Normal = compute_normals(newsnake)
        Fexternal, update_normal = update_normals_Fext(newsnake, image)
        test = np.diag(Fexternal) @ update_normal
        #normals = update_normals(Normal, mean_inside, mean_outside, mean, contour, image)
        newsnake = (newsnake + ((delta_t * np.diag(Fexternal)) @ update_normal))
        Bint = backward_euler_Bint(newsnake, alpha, beta, delta_t)
        newsnake = Bint @ newsnake 

            #Inside your iteration loop:
        if i % 100 == 0:
            newsnake = resample_contour(newsnake, spacing=10)
            newsnake = enforce_bounds(newsnake, image.shape)

        if i % 200 == 0:
            Fexternal, update_normal = update_normals_Fext(newsnake, image)
            test = np.diag(Fexternal) @ update_normal
            fig, axes = plt.subplots(2, 1)

            axes[0].imshow(image, cmap='gray')
            axes[0].scatter(newsnake[:, 0], newsnake[:, 1], color='red', s=5) 
            axes[0].plot(newsnake[:, 0], newsnake[:, 1]) 
            axes[0].quiver(
                newsnake[:, 0],  # x-coordinates of the points
                newsnake[:, 1],  # y-coordinates of the points
                test[:, 0],  # x-components of the normals
                test[:, 1],  # y-components of the normals
                angles='xy', scale_units='xy', scale=0.2, color='green', label='Normals'
            )
            mean_inside, mean_outside, mean = compute_mean_intensities(image, newsnake)
            mean_inside = mean_inside
            mean_outside = mean_outside
            mean = mean
            # Call the function to plot the snake points
            snake_values = get_pixel_values_at_snake_points(image, newsnake)
            axes[1].plot(snake_values)
            # Add a horizontal line at y = 11
            axes[1].axhline(y=mean_inside, color='r', linestyle='--')
            axes[1].axhline(y=mean_outside, color='r', linestyle='--')
            axes[1].axhline(y=mean, color='r', linestyle='-')
            # Adjust layout
            plt.tight_layout()
            plt.title(f"Iteration {i}: α={alpha}, β={beta}, Δt={delta_t}")
            # Show the plots
            plt.show(block=False)
            plt.pause(pause_time)
            plt.close("all")


    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.scatter(newsnake[:, 0], newsnake[:, 1], color='red', s=1) 
    plt.plot(newsnake[:, 0], newsnake[:, 1]) 
    plt.quiver(
        newsnake[:, 0],  # x-coordinates of the points
        newsnake[:, 1],  # y-coordinates of the points
        test[:, 0],  # x-components of the normals
        test[:, 1],  # y-components of the normals
        angles='xy', scale_units='xy', scale=0.2, color='green', label='Normals'
    )
    mean_inside, mean_outside, mean = compute_mean_intensities(image, newsnake)
    mean_inside = mean_inside
    mean_outside = mean_outside
    mean = mean
    # Call the function to plot the snake points
    #snake_values = get_pixel_values_at_snake_points(image, newsnake)
    #axes[1].plot(snake_values)
    # Add a horizontal line at y = 11
    #axes[1].axhline(y=mean_inside, color='r', linestyle='--')
    #axes[1].axhline(y=mean_outside, color='r', linestyle='--')
    #axes[1].axhline(y=mean, color='r', linestyle='-')
    # Adjust layout
    plt.tight_layout()
    plt.title(f"Iteration {i}: α={alpha}, β={beta}, Δt={delta_t}")
    # Show the plots
    plt.show(block=False)
    plt.pause(pause_time)
    plt.close("all")

    return newsnake


######################################
def main():
    # Example usage
    # Create a synthetic image with a bright square inside a dark background
    label, data, path = load_input()
    
    alpha = 0.15
    beta = 0.3
    delta_t = 0.5
    itterations =  200 
    pause_time = 1
    
    if path is True:
        image = data/255

        # Define an initial contour (snake)
        contour = create_circle_snake(image, 100, 100)
        
        # Set the initial sanke values and 
        #Fexternal, update_normal = update_normals_Fext(contour, image)
        #Bint = backward_euler_Bint(contour, alpha, beta, delta_t)
        #newsnake = Bint @ (contour + ((delta_t * np.diag(Fexternal)) @  update_normal))
        newsnake = contour
        newestsnake = Active_Contour(alpha, beta, delta_t, newsnake, image, itterations, pause_time)

        # info for plotting the final output compared to input with Normal Vectors 
        Fexternal, update_normal = update_normals_Fext(newestsnake, image)
        Corrrrect_dir_normal = np.diag(Fexternal) @ update_normal

        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.scatter(contour[:, 0], contour[:, 1], color='red', s=1) 
        plt.plot(newestsnake[:, 0], newestsnake[:, 1]) 
        plt.quiver(
            newestsnake[:, 0],  # x-coordinates of the points
            newestsnake[:, 1],  # y-coordinates of the points
            Corrrrect_dir_normal[:, 0],  # x-components of the normals
            Corrrrect_dir_normal[:, 1],  # y-components of the normals
            angles='xy', scale_units='xy', scale=2, color='green', label='Normals'
        )
        mean_inside, mean_outside, mean = compute_mean_intensities(image, contour)
        mean_inside = mean_inside
        mean_outside = mean_outside
        mean = mean

        plt.tight_layout()
        plt.title(f"Final: α={alpha}, β={beta}, Δt={delta_t}")
        # Show the plots
        plt.show()

    elif path is False:
        video = data
        # Define an initial contour (snake)
        snakes = []
        
        for i, frame in enumerate(video):
            if i == 0:
                print(frame.shape)
                contour = create_circle_snake(frame, 40, 40)
                newsnake = contour
            #if i == 100:
            #    image = frame/255
            #    plt.figure()
            #    plt.imshow(image, cmap='gray')
            #    plt.show()
            
            print(f"Frame {i} shape:", frame.shape)

            image = frame/255
            

            # Set the initial sanke values and 
            #Fexternal, update_normal = update_normals_Fext(contour, image)
            #Bint = backward_euler_Bint(contour, alpha, beta, delta_t)
            #newsnake = Bint @ (contour + ((delta_t * np.diag(Fexternal)) @  update_normal))
            newestsnake = Active_Contour(alpha, beta, delta_t, newsnake, image, itterations, pause_time)
            #snakes.append(newestsnake)
            

            Fexternal, update_normal = update_normals_Fext(newestsnake, image)
            Corrrrect_dir_normal = np.diag(Fexternal) @ update_normal

            plt.figure()
            plt.imshow(image, cmap='gray')
            plt.scatter(newsnake[:, 0], newsnake[:, 1], color='red', s=1) 
            plt.plot(newestsnake[:, 0], newestsnake[:, 1]) 
            plt.quiver(
                newestsnake[:, 0],  # x-coordinates of the points
                newestsnake[:, 1],  # y-coordinates of the points
                Corrrrect_dir_normal[:, 0],  # x-components of the normals
                Corrrrect_dir_normal[:, 1],  # y-components of the normals
                angles='xy', scale_units='xy', scale=0.2, color='green', label='Normals'
            )
            mean_inside, mean_outside, mean = compute_mean_intensities(image, contour)
            mean_inside = mean_inside
            mean_outside = mean_outside
            mean = mean

            plt.tight_layout()
            plt.title(f"Final: α={alpha}, β={beta}, Δt={delta_t}")
            # Show the plots
            plt.show(block=False)
            plt.pause(pause_time)
            plt.close("all")
            newsnake = newestsnake
            
            
def save_video():
    label, data, path = load_input()
    video = data
    for i, frame in enumerate(video):
        image = frame
        img = Image.fromarray(image)
        img = img.convert("L")

        # Save the image as a PNG file
        img.save("my_image.png")
        
        
        plt.figure()
        
        plt.imshow(image)
        plt.savefig('my_plot.png', bbox_inches='tight', pad_inches=0)
        plt.show()
        
        print(image)
        # Create an Image object from the array

        
#save_video()
main()