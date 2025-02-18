import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.animation as animation
import cv2 as cv
import tkinter as tk
from tkinter import Image, filedialog
from PIL import Image
from shapely.geometry import LineString
from scipy.ndimage import sobel
import skimage
from skimage.draw import polygon

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
        # read the image in grayscale
        img_grayscale = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
        if img_grayscale is None:
            raise Exception("ERROR: CV Unable to load image file")
        

    except Exception as e:
        # if OpenCV image input can't be read, display an error message 
        print(e)
        
        try:
            # Try and take the input from text (Used for Part 1 Smoothing using Euler)
            img_grayscale = np.loadtxt(input_file)
            
            if img_grayscale is None:
                raise Exception("ERROR: numpy not able to load image file")
        
        except Exception as e:
            # if txt file cant be read, display an error message 
            print(e)

            try:
                #Try and read it as a video if it isnt taken in as an image in the last 2 checks  
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

def compute_mean(image, contour):
    """
    Compute the mean intensity inside and outside the contour.
    :param image: A 2D numpy array representing the image.
    :param contour: A numpy array of shape (N, 2) representing the (x, y) coordinates of the contour.
    :return: A tuple (mean_inside, mean_outside) representing the mean intensities.
    """
    # Create a mask for the region inside the contour
    mask = skimage.draw.polygon2mask(image.shape, contour[:, ::-1])
    closed_contour = np.vstack([contour, contour[0]])  # Ensure the contour is closed
    rr, cc = polygon(closed_contour[:, 1], closed_contour[:, 0], image.shape)
    mask = np.zeros_like(image, dtype=bool)
    mask[rr, cc] = True
    
    # Create masked arrays for inside and outside the contour
    image_inside = np.ma.masked_where(~mask, image)  # Mask outside the contour
    image_outside = np.ma.masked_where(mask, image)  # Mask inside the contour
    
    # Compute mean intensities
    mean_inside = image_inside.mean() if image_inside.count() > 0 else np.nan
    mean_outside = image_outside.mean() if image_outside.count() > 0 else np.nan
    
    # Compute the overall mean of the image
    mean = np.mean(image)
    
    #plt.imshow(image, cmap='gray')
    #plt.imshow(image_outside)
    
    #plt.show()
    
    return mean_inside, mean_outside, mean

def compute_normals(contour):
    """
    Compute the normal vectors for a 2D contour using the point before and after.
    :param contour: A numpy array of shape (N, 2) representing the (x, y) coordinates of the contour.
    :return: A numpy array of shape (N, 2) representing the normal vectors at each point.
    """
    # Compute tangent vectors using the point before and after
    tangent = np.roll(contour, -1, axis=0) - np.roll(contour, 1, axis=0)
    
    # Compute normal vectors by rotating tangent vectors by 90 degrees
    normal = np.zeros_like(tangent)
    normal[:, 0] = -tangent[:, 1]  # N_x = -T_y
    normal[:, 1] = tangent[:, 0]   # N_y = T_x
    
    # Normalize the normal vectors
    magnitude = np.linalg.norm(normal, axis=1, keepdims=True)
    magnitude[magnitude == 0] = 1  # Avoid division by zero
    unit_normal = normal / magnitude

    return -unit_normal

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
    image = image
    normals = compute_normals(snake)
    values_at_snake = get_pixel_values_at_snake_points(image, snake)
    
    #mean_inside, mean_outside, mean = compute_mean_intensities(image, snake)
    mean_inside, mean_outside, known = compute_mean(image, snake)
    if np.any(np.isnan(mean_inside)):
        raise ValueError("outside contains NaN values.")
    if np.any(np.isnan(mean_outside)):
        raise ValueError("outside contains NaN values.")
    Fext = 2 *(mean_inside - mean_outside) * (((values_at_snake) - ((1/2)*(mean_inside + mean_outside))))/255
    
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
    
    #print("D1:", D1)
    #print("D2:", D2)
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

def detect_loops(contour):
    """
    Detect loops in the snake contour.
    :param contour: A numpy array of shape (N, 2) representing the snake points.
    :return: List of indices where loops occur.
    """
    loop_indices = []
    n = len(contour)
    
    for i in range(n):
        # Create a line segment from point i to i+1
        line1 = LineString([contour[i], contour[(i+1)%n]])
        
        for j in range(i+2, n-1):
            # Create a line segment from point j to j+1
            line2 = LineString([contour[j], contour[(j+1)%n]])
            
            # Check if the two line segments intersect
            if line1.intersects(line2):
                loop_indices.append((i, j))
    
    return loop_indices

def resolve_loops(contour):
    """
    Resolve loops in the snake contour.
    :param contour: A numpy array of shape (N, 2) representing the snake points.
    :return: A new contour with loops resolved.
    """
    loop_indices = detect_loops(contour)
    
    if not loop_indices:
        return contour  # No loops detected
    
    # Resolve the first detected loop
    i, j = loop_indices[0]
    
    # Remove the loop by reconnecting the snake
    new_contour = np.vstack([contour[:i+1], contour[j+1:]])
    
    return new_contour

def Active_Contour(alpha, beta, delta_t, newsnake, image, itterations, pause_time, gifitter):
    
    for i in range(0, itterations):
       
        Fexternal, update_normal = update_normals_Fext(newsnake, image)
        newestsnake = (newsnake + (delta_t * (np.diag(Fexternal) @ update_normal)))
        Bint = backward_euler_Bint(newestsnake, alpha, beta, delta_t)
        newestsnake = Bint @ newestsnake
        
        newestsnake = resolve_loops(newestsnake)
        
        if i % 10 == 0:
            newestsnake = resample_contour(newestsnake, spacing=15)
            newestsnake = enforce_bounds(newestsnake, image.shape)
        
        #if i % 10 == 0:
            
        Fexternal, update_normal = update_normals_Fext(newestsnake, image)
        test = np.diag(Fexternal) @ update_normal
        
        fig, axes = plt.subplots(2, 1)
        axes[0].imshow(image, cmap='gray')
        axes[0].scatter(newestsnake[:, 0], newestsnake[:, 1], color='red', s=5) 
        axes[0].plot(newestsnake[:, 0], newestsnake[:, 1]) 
        axes[0].quiver(
            newestsnake[:, 0],  # x-coordinates of the points
            newestsnake[:, 1],  # y-coordinates of the points
            test[:, 0],  # x-components of the normals
            test[:, 1],  # y-components of the normals
            angles='xy', scale_units='xy', scale=0.5, color='green', label='Normals'
        )
        mean_inside, mean_outside, mean = compute_mean(image, newestsnake)
        
        # Call the function to get values at snake points for plotting
        snake_values = get_pixel_values_at_snake_points(image, newestsnake)
        axes[1].plot(snake_values)
        axes[1].axhline(y=mean_inside, color='r', linestyle='--')
        axes[1].axhline(y=mean_outside, color='r', linestyle='--')
        axes[1].axhline(y=mean, color='r', linestyle='-')
        # Adjust layout
        plt.tight_layout()
        plt.title(f"Iteration {i}: α={alpha}, β={beta}, Δt={delta_t}")
        # Show the plots
            # Save with dynamic filename
        #filename = f"gifpngs/image_{gifitter}{i}.png"  # Automatically updates the filename with the loop index
        #plt.savefig(filename, format="png")
        #plt.clf()  # Clear the figure to avoid overwriting the same plot
        plt.show()#plt.show(block=False)
        #plt.pause(pause_time)
        #plt.close("all")
            
        newsnake = newestsnake

    #Fexternal, update_normal = update_normals_Fext(newestsnake, image)
    #test = np.diag(Fexternal) @ update_normal
    #
    #plt.figure()
    #plt.imshow(image, cmap='gray')
    #plt.scatter(newsnake[:, 0], newsnake[:, 1], color='red', s=1) 
    #plt.plot(newsnake[:, 0], newsnake[:, 1]) 
    #plt.quiver(
    #    newsnake[:, 0],  # x-coordinates of the points
    #    newsnake[:, 1],  # y-coordinates of the points
    #    test[:, 0],  # x-components of the normals
    #    test[:, 1],  # y-components of the normals
    #    angles='xy', scale_units='xy', scale=0.2, color='green', label='Normals'
    #)
    #mean_inside, mean_outside, mean = compute_mean(image, newsnake)
    #mean_inside = mean_inside
    #mean_outside = mean_outside
    #mean = mean
    ## Call the function to plot the snake points
    #snake_values = get_pixel_values_at_snake_points(image, newsnake)
    #axes[1].plot(snake_values)
    #
    #axes[1].axhline(y=mean_inside, color='r', linestyle='--')
    #axes[1].axhline(y=mean_outside, color='r', linestyle='--')
    #axes[1].axhline(y=mean, color='r', linestyle='-')
    ## Adjust layout
    #plt.tight_layout()
    #plt.title(f"Iteration {i}: α={alpha}, β={beta}, Δt={delta_t}")
    ## Show the plots
    #plt.show()
    #plt.pause(pause_time)
    #plt.close("all")
    
    return newsnake

def main():
    
    label, data, path = load_input()
    
    alpha = 0.15
    beta = 0.3
    delta_t = 0.05
    itterations = 75
    pause_time = 1
    
    if path is True:
        image = data

        # Define an initial contour (snake)
        contour = create_circle_snake(image, 100, 100)
        
        # Set the initial sanke values and compute new snake
        newsnake = contour
        newestsnake = Active_Contour(alpha, beta, delta_t, newsnake, image, itterations, pause_time, 0)

        # info for plotting the final output compared to input with Normal Vectors 
        Fexternal, update_normal = update_normals_Fext(newestsnake, image)
        Corrrrect_dir_normal = update_normal

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
        mean_inside, mean_outside, mean = compute_mean(image, contour)

        plt.tight_layout()
        plt.title(f"Final: α={alpha}, β={beta}, Δt={delta_t}")
        # Show the plots
        plt.show()

    elif path is False:
        video = data
        # Define an initial contour (snake)
        snakes = []
        giffitter = 0
        for i, frame in enumerate(video):
            if i == 0:
                print(frame.shape)
                contour = create_circle_snake(frame, 100, 100)
                newsnake = contour
            
            print(f"Frame {i} shape:", frame.shape)

            image = frame
            
            #An array where the most relative moment happens 5,35,40,50,135,140,155,160,165,255,260
            movment_frames = np.array([1,25,130,145,150,170,240,245,250])
            if i in movment_frames:
                newestsnake = Active_Contour(alpha, beta, delta_t, newsnake, image, itterations, pause_time, giffitter)
                newsnake = newestsnake
                giffitter += 1
            



def save_image():
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
 
def test():
    
    label, data, path = load_input() 

#test()  
#save_image()
main()