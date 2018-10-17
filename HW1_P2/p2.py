import numpy as np
from math import sqrt, pi, sin, cos
from cv2 import line

def find_edge(gray_in, threshold):
    height = len(gray_in)
    width = len(gray_in[0])
    v_filter = [-1,-2,-1,
                 0, 0, 0,
                 1, 2, 1]
    h_filter = [-1, 0, 1,
                -2, 0, 2,
                -1, 0, 1]
    v_edges = np.zeros((height, width))
    h_edges = np.zeros((height, width))
    edges   = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            filter_ind = 0
            for y_w in range(y-1,y+2):
                if y_w < 0:
                    y_w = 0
                if y_w >= height:
                    y_w = height-1
                for x_w in range(x-1,x+2):
                    if x_w < 0:
                        x_w = 0
                    if x_w >= width:
                        x_w = width-1
                    v_edges[y][x] += gray_in[y_w][x_w] * v_filter[filter_ind]
                    h_edges[y][x] += gray_in[y_w][x_w] * h_filter[filter_ind]
                    filter_ind += 1

    for y in range(height):
        for x in range(width):
            edges[y][x] = sqrt(v_edges[y][x]**2 + h_edges[y][x]**2)

    thresholded_edge_img = np.where(edges > threshold, 255, 0)
    return thresholded_edge_img

def hough(edge_in, theta_nbin, rho_nbin):
    accumulator = np.zeros((rho_nbin, theta_nbin))

    theta_max = pi/2
    theta_min = -1 * theta_max

    rho_max = sqrt(len(edge_in)**2 + len(edge_in[0])**2)
    rho_min = -1 * rho_max
    rho_unit = (rho_max - rho_min)/rho_nbin

    for y in range(len(edge_in)):
        for x in range(len(edge_in[0])):
            if edge_in[y][x] > 0:
                theta_i = 0
                for theta_0 in np.linspace(theta_min, theta_max, theta_nbin):
                    rho = y * cos(theta_0) - x * sin(theta_0)
                    if (not rho < rho_min) and (not rho > rho_max):
                        accumulator[int((rho+rho_max)/rho_unit)][theta_i] += 0.25
                    theta_i += 1

    accumulator = accumulator
    hough_res = np.clip(accumulator, 0, 255)
    return hough_res

def hough_line(gray_in, accumulator_array, hough_threshold):
    grey_out_with_edge = gray_in.copy()

    theta_max = pi/2
    theta_min = -1 * theta_max
    theta_unit = (theta_max - theta_min)/len(accumulator_array[0])

    rho_max = sqrt(len(gray_in)**2 + len(gray_in[0])**2)
    rho_min = -1 * rho_max
    rho_unit = (rho_max - rho_min)/len(accumulator_array)

    x_max = len(gray_in[0])-1
    y_max = len(gray_in)-1
 

    for rho in range(len(accumulator_array)):
        for theta in range(len(accumulator_array[0])):
            if accumulator_array[rho][theta] > hough_threshold:
                r = rho*rho_unit - rho_max
                t = theta*theta_unit - theta_max
                x_1 = -1
                x_2 = -1
                y_1 = -1
                y_2 = -1
                # Find endpoints
                for x in range(len(gray_in[0])):
                    y = int((x*sin(t) + r)/cos(t))
                    if x_1 == -1:
                        if y >= 0 and y <= y_max:
                            x_1 = x
                            y_1 = y
                    else:
                        if y >= 0 and y <= y_max:
                            x_2 = x
                            y_2 = y
                line(grey_out_with_edge, (x_1,y_1), (x_2,y_2), (250,0,0))
    return grey_out_with_edge
