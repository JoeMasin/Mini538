import numpy as np
import scipy
import skimage
import matplotlib as plt




########################################
"""
Part 2 - Image Segmentation using Active Contours
"""

# load image to be segmented
#plusplus = cv.imread("./images/plusplus.png", cv.IMREAD_GRAYSCALE)
#img = cv.imread("./images/sugar_2.png", cv.IMREAD_GRAYSCALE)
# img = ~cv.equalizeHist(img)


def create_contour(radius, num_points, center):
    """
    Create a circular contour

    :param radius: radius of contour
    :param num_points: number of points in the contour
    :param center: center of the list of points
    :return: list of points in the contour
    """

    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)

    return np.column_stack((x, y))


def snakes_segmentation(img, num_pts, tau, alpha, beta, iters):
    """
    Performs snakes segmentation on in input image

    :param img: input image
    :param num_pts: number of points in the segmentation contour
    :param tau: time step
    :param alpha: elasticity
    :param beta: rigidity
    :param iters: number of iterations
    :return: segmented image
    """

    # extract info from image
    height, width = img.shape
    radius = width / 4
    center = (width / 2, height / 2)

    # create initial contour
    contour_pts = create_contour(radius, num_pts, center)

    # start pyplot GUI
    plt.ion()
    figure, ax = plt.subplots(figsize=(5, 4))

    # get x and y coordinates from points in contour
    x_pts = contour_pts[:, 0]
    y_pts = contour_pts[:, 1]

    # plot image
    ax.imshow(img, cmap='gray')

    # draw contour
    contour, = ax.plot(x_pts, y_pts, 'r')

    for iteration in range(iters):

        # create a mask of the image with the contour
        mask = skimage.draw.polygon2mask(img.shape, contour_pts[:, ::-1])

        f_ext = []
        for pt in contour_pts:

            # compute mean pixel intensities inside & outside the contour
            m_in = img[mask].mean()
            m_out = img[~mask].mean()

            # compute the external force using the mean pixel intensities
            f_ext.append(2 * (m_in - m_out) * (img[(int(pt[1]), int(pt[0]))] - 1/2 * (m_in + m_out)) / 255)

        f_ext = np.array(f_ext)

        # calculate directions for each point in contour
        tangents = np.roll(contour_pts, -1, axis=0) - np.roll(contour_pts, 1, axis=0)

        # convert directions into unit normal vectors
        normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        N = normals / norms

        # get new contour pts
        contour_pts_new = contour_pts + tau * np.diag(f_ext) @ N


        contour_pts_new = smooth_backward_euler(contour_pts_new, alpha, beta, 1, 0)
        # contour_pts_new = smooth_forward_euler(contour_pts_new, alpha, beta, 10, 0)

        # update values
        contour.set_xdata(contour_pts_new[:, 0])
        contour.set_ydata(contour_pts_new[:, 1])
        contour_pts = contour_pts_new

        # update GUI
        figure.canvas.draw()
        plt.title(f"Snakes Segmentation\nAlpha = {alpha}, Beta = {beta}, Tau = {tau}, Iteration {iteration + 1}")
        figure.canvas.flush_events()

    ## Turn off interactive mode after the loop
    plt.ioff()
    plt.show()


snakes_segmentation(img, 200, 0.03, 0.15, 0.3, 100)