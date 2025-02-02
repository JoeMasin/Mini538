#import pandas
#import scipy 
#import matplotlib
#import math
from types import new_class
from unittest import result
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import filedialog

###################################### Know it works below
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

### The internal component  is relient on the snake aka the current contour that we are deforming to match the "object" in the image 
### The extrenal energy is relient on the input image that we will use to deforme the snake 
def internal_energy(snake, alpha, beta): # 
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

def internal_force(snake, alpha, beta): # 
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

def backward_euler(snake, alpha, beta, delta_t): #  Fixed to work correctly 
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
    
    # Solve for c_new
    snake_updated = BE_matrix @ snake
    
    return snake_updated

def forward_euler(snake, alpha, beta, delta_t): #  Fixed to work correctly
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
    N = len(snake)
    I = np.eye(N)  # Identity matrix
    
    # First-order finite difference matrix (elasticity)
    D1 =  np.roll(I, -1, axis=0) - 2 * I + np.roll(I, 1, axis=0)
    
    # Second-order finite difference matrix (bending)
    D2 = np.roll(I, -2, axis=0) - 4 * np.roll(I, -1, axis=0) + 6 * I - 4 * np.roll(I, 1, axis=0) + np.roll(I, 2, axis=0)
    
    # Internal force matrix
    A = alpha * D1 + beta * D2
    
    # Forward Euler matrix
    FE_matrix = I - delta_t * A
    
    # Solve for c_new
    snake_updated = FE_matrix @ snake
    
    return snake_updated

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

def forward_euler_Bint(snake, alpha, beta, delta_t): #  Fixed to work correctly
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
    N = len(snake)
    I = np.eye(N)  # Identity matrix
    
    # First-order finite difference matrix (elasticity)
    D1 =  np.roll(I, -1, axis=0) - 2 * I + np.roll(I, 1, axis=0)
    
    # Second-order finite difference matrix (bending)
    D2 = np.roll(I, -2, axis=0) - 4 * np.roll(I, -1, axis=0) + 6 * I - 4 * np.roll(I, 1, axis=0) + np.roll(I, 2, axis=0)
    
    # Internal force matrix
    A = alpha * D1 + beta * D2
    
    # Forward Euler matrix
    FE_matrix = I - delta_t * A
    
    return FE_matrix

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
#####################################

"""def external_force(snake, F_internal, E_external):
    ###
    Compute the external force for the snake.

    Parameters:
    snake: ndarray of shape (N, 2)
        Coordinates of the snake's points [(x1, y1), (x2, y2), ...].
    F_internal: ndarray of shape (N, 2)
        Internal forces acting on each point of the snake.
    E_external: ndarray of shape (H, W)
        External energy field (e.g., image gradients or boundary energies).

    Returns:
    F_external: ndarray of shape (N, 2)
        External forces acting on each point of the snake.
    ###
    N = len(snake)
    H, W = E_external.shape  # Dimensions of the external energy field
    
    # Gradient of the external energy field (negative gradient for force)
    grad_x, grad_y = np.gradient(-E_external)
    
    # Map external force from the energy field to each snake point
    F_external = np.zeros_like(snake)
    for i in range(N):
        x, y = snake[i]
        # Clamp snake positions to the image boundaries
        x_clamped = max(0, min(int(x), W - 1))
        y_clamped = max(0, min(int(y), H - 1))
        
        F_external[i, 0] = grad_x[y_clamped, x_clamped]
        F_external[i, 1] = grad_y[y_clamped, x_clamped]
    
    # Compute mean of internal and external forces
    mean_internal = np.mean(F_internal, axis=0)
    mean_external = np.mean(F_external, axis=0)
    
    # Compute the external force
    I = np.eye(2)  # Identity matrix
    scaling_matrix = I - 0.5 * (mean_internal + mean_external)
    external_component = 2 * (mean_internal - mean_external)
    
    F_adjusted_external = external_component @ scaling_matrix
    
    # Apply the computed external adjustment to each snake point
    for i in range(N):
        F_external[i] += F_adjusted_external
    
    return F_external
    """

def external_energy(image): # 
    """
    Compute the external energy for the snake algorithm, which is based on the negative gradient magnitude of the image.
    
    Parameters:
    - image_path: str, path to the input image
    
    Returns:
    - external_energy_normalized: np.array, normalized external energy for visualization
    """
    
    # Step 2: Compute image gradients using Sobel filter (X and Y directions)
    gradient_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)  # Gradient in the x direction
    gradient_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)  # Gradient in the y direction
    
    # Step 3: Compute the gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Step 4: Calculate external energy (negative gradient magnitude)
    external_energy = -gradient_magnitude
    
    # Step 5: Normalize the external energy for better visualization (optional)
    external_energy_normalized = cv.normalize(external_energy, None, 0, 255, cv.NORM_MINMAX)
    
    return external_energy_normalized

def external_force(image):
    """
    Compute the external force for the snake algorithm based on the negative gradient of the external energy.

    Parameters:
    - image
    
    Returns:
    - external_force_x: np.array, external force in the x direction
    - external_force_y: np.array, external force in the y direction
    """
    
    # Step 2: Compute image gradients using Sobel filter (X and Y directions)
    gradient_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)  # Gradient in the x direction
    gradient_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)  # Gradient in the y direction
    
    # Step 3: Compute the gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Step 4: Calculate external energy (negative gradient magnitude)
    external_energy = -gradient_magnitude
    
    # Step 5: Compute the gradient of external energy (external force)
    # Using Sobel filters again to get the gradient of external energy
    external_force_x = cv.Sobel(external_energy, cv.CV_64F, 1, 0, ksize=3)  # Force in the x direction
    external_force_y = cv.Sobel(external_energy, cv.CV_64F, 0, 1, ksize=3)  # Force in the y direction
    
    # The external force is the negative gradient of the gradient magnitude
    force_x = -external_force_x  # Force in the x direction
    force_y = -external_force_y  # Force in the y direction
    

    return force_x, force_y

    #return external_force_x, external_force_y

def get_pixel_values_at_snake_points(image, snake_points):
    """
    Extracts the pixel values from the image at the locations specified by the snake points.
    
    Parameters:
    - image: np.ndarray, input image (height x width, or height x width x 3 for color).
    - snake_points: np.ndarray, snake points (Nx2 array, where N is the number of points).
    
    Returns:
    - pixel_values: np.ndarray, extracted pixel values at the snake points.
    """
    pixel_values = []
    
    for point in snake_points:
        y, x = point  # Get the (y, x) coordinates
        pixel_value = image[int(y), int(x)]  # Extract the pixel value at that location
        pixel_values.append(pixel_value)
        out = np.array(pixel_values)/255
    
    return out 

#def updateSnake(snake, internal_forces, external_forces, )

def calculate_mean_intensities(snake_coords, image):
    # Load image and convert to grayscale (if not already)
    
    
    # Create a grid of coordinates
    h, w = image.shape
    y, x = np.mgrid[:h, :w]
    points = np.vstack((x.ravel(), y.ravel())).T  # (width*height, 2)
    
    # Define the snake as a closed polygon
    snake_path = Path(snake_coords, closed=True)
    
    # Determine which points are inside the snake
    inside_mask = snake_path.contains_points(points).reshape(h, w)
    
    # Calculate mean intensity for inside and outside
    inside_mean = image[inside_mask].mean()
    outside_mean = image[~inside_mask].mean()
    
    return inside_mean, outside_mean

def compute_mean_in_boundary(image, boundary):
    """
    Compute the mean value of an image inside a polygonal boundary.

    Parameters:
    - image: 2D (grayscale) or 3D (color) numpy array representing the image.
    - boundary: Nx2 numpy array of (x, y) coordinates defining the polygon.

    Returns:
    - mean_value: Mean pixel value inside the boundary.
    """
    # Create an empty mask with the same dimensions as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Fill the polygon defined by the boundary with white (255)
    cv.fillPoly(mask, [boundary.astype(np.int32)], 255)

    # Use the mask to extract the region from the image
    masked_region = cv.bitwise_and(image, image, mask=mask)

    # Compute the mean value of the non-zero region
    mean_value_in = cv.mean(image, mask=mask)
    mean_value_out = cv.mean(image, mask=cv.bitwise_not(mask))
    mean = cv.mean(image)

    # If the image is grayscale, return the first value; if color, return all channels
    return mean_value_in, mean_value_out, mean

def external_force_N(image, snake, delta_t): #  Fixed to work correctly
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
    N = len(snake)
    I = np.eye(N)
    meanin, meanout, mean = compute_mean_in_boundary(image, snake)
    meanin = meanin[0]/255
    meanout = meanout[0]/255
    Fext = (I - (1/2)*(meanin + meanout))
    
    return Fext

def compute_normals(contour):
    """
    Compute the normal vectors for a 2D contour.
    :param contour: A numpy array of shape (N, 2) representing the (x, y) coordinates of the contour.
    :return: A numpy array of shape (N, 2) representing the normal vectors at each point.
    """
    # Compute tangent vectors using central differences
    tangent = np.roll(contour, -1, axis=0) - np.roll(contour, 1, axis=0)
    
    # Compute normal vectors by rotating tangent vectors by 90 degrees
    normal = np.zeros_like(tangent)
    normal[:, 0] = -tangent[:, 1]  # N_x = -T_y
    normal[:, 1] = tangent[:, 0]   # N_y = T_x
    
    # Normalize the normal vectors
    magnitude = np.linalg.norm(normal, axis=1, keepdims=True)
    magnitude[magnitude == 0] = 1  # Avoid division by zero
    unit_normal = normal / magnitude
    
    return unit_normal

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

def Test_Fint(): #
    #Read in and plot the initial data
    Input_FileName, Inital_data = load_input()
    plt.plot(Inital_data[:,0], Inital_data[:,1])
    plt.show()
    
    # Set the initial peramiters to be used for alpha, beta, and the time step 
    alpha = 0.05  # Elasticity weight
    beta = 0.3   # Bending weight
    delta_t = 0.2 # Time step

    #Print the internal energy of the data
    energy = internal_energy(Inital_data, alpha, beta)
    print(energy)

    # Plot internal forces as arrows
    F_internal = internal_force(Inital_data, alpha, beta)
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

    # Update snake using Backward Euler
    snake_updated = backward_euler(Inital_data, alpha, beta, delta_t)
    
    # Plot the original and updated snake
    plt.figure()
    plt.plot(Inital_data[:, 0], Inital_data[:, 1], label="Original Snake")
    plt.plot(snake_updated[:, 0], snake_updated[:, 1], label="Updated Snake")
    plt.title("Snake Evolution (Backward Euler)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

    # Update snake using Forward Euler
    snake_updated = forward_euler(Inital_data, alpha, beta, delta_t)
    # Plot the original and updated snake
    plt.figure()
    plt.plot(Inital_data[:, 0], Inital_data[:, 1], label="Original Snake")
    plt.plot(snake_updated[:, 0], snake_updated[:, 1], label="Updated Snake")
    plt.title("Snake Evolution (Forward Euler)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

def main():
    Input_FileName, Inital_data = load_input()
    
    # Set the initial peramiters to be used for alpha, beta, and the time step 
    alpha = 0.1  # Elasticity weight
    beta = 0.3   # Bending weight
    delta_t = 0.1 # Time step
    
    #external_energy_img = external_energy(Inital_data)
    ### Plot the result
    ##plt.figure(figsize=(10, 5))
    #plt.imshow(external_energy_img, cmap='hot')
    #plt.title('External Energy (Negative Gradient)')
    #plt.axis('off')
    #plt.show()
    
    image = Inital_data
    radius = 200 #tested to be a decent starting point for the "snake"
    snake = create_circle_snake(image, radius, 100)

    meanin, meanout, mean = compute_mean_in_boundary(image, snake)
    print(meanin)
    print(meanout)
    print(mean)
    
    fig, axes = plt.subplots(2, 1)

    axes[0].imshow(image, cmap='gray')
    axes[0].scatter(snake[:, 0], snake[:, 1], color='red', s=1)

    # Call the function to plot the snake points
    snake_values = get_pixel_values_at_snake_points(image, snake)
    axes[1].plot(snake_values)
    # Add a horizontal line at y = 11
    axes[1].axhline(y=meanin[0], color='r', linestyle='--')
    axes[1].axhline(y=meanout[0], color='r', linestyle='--')
    axes[1].axhline(y=mean[0], color='r', linestyle='-')
    # Adjust layout
    plt.tight_layout()
    # Show the plots
    plt.show()
    Fext = external_force_N(image, snake, delta_t)
    
    Bint = forward_euler_Bint(snake, alpha, beta, delta_t) 
    
    Normal = compute_normals(snake)
    
    newsnake = Bint @ (snake + (delta_t * Fext @ Normal))
    
    meanin, meanout, mean = compute_mean_in_boundary(image, snake)
    normalstoplot = Fext @ Normal

    visualize_contour_with_normals(snake, normalstoplot, scale=0.05)
    
    for i in range(0, 100):
        Fext = external_force_N(image, newsnake, delta_t)
        
        Bint = forward_euler_Bint(newsnake, alpha, beta, delta_t) 
        
        Normal = compute_normals(newsnake)
        
        newsnake = Bint @ (newsnake + (delta_t * np.diagonal(Fext) @ Normal))
        
        fig, axes = plt.subplots(2, 1)

        axes[0].imshow(image, cmap='gray')
        axes[0].scatter(newsnake[:, 0], newsnake[:, 1], color='red', s=1) 

        # Call the function to plot the snake points
        snake_values = get_pixel_values_at_snake_points(image, newsnake)
        axes[1].plot(snake_values)
        # Add a horizontal line at y = 11
        axes[1].axhline(y=meanin[0]/255, color='r', linestyle='--')
        axes[1].axhline(y=meanout[0]/255, color='r', linestyle='--')
        axes[1].axhline(y=mean[0]/255, color='r', linestyle='-')
        # Adjust layout
        plt.tight_layout()
        # Show the plots
        plt.show()
        
    
main()
#Test_Fint()

