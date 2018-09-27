import numpy as np
from scipy.misc import imread

def read_image(im_path):
    return imread(im_path)

def histogram(gray_in):
    hist = np.zeros(256)
    for h in range(len(gray_in)):
        for w in range(len(gray_in[h])):
            hist[gray_in[h][w]] += 1
    return hist

def denoisy_median_filtering(gray_in, diameter=3):
    denoised_img = np.zeros((len(gray_in),len(gray_in[0])))
    for h in range(len(gray_in)):
        for w in range(len(gray_in[0])):
            window = np.zeros(diameter*diameter)
            radius = int((diameter - 1)/2)
            i = 0

            for h_w in range(h-radius,h+radius+1):
                if h_w < 0:
                    h_w = 0
                elif h_w >= len(gray_in):
                    h_w = len(gray_in) - 1

                for w_w in range(w-radius,w+radius+1):
                    if w_w < 0:
                        w_w = 0
                    elif w_w >= len(gray_in[0]):
                        w_w = len(gray_in[0]) - 1
                    window[i] = gray_in[h_w][w_w]
                    i += 1


            denoised_img[h][w] = np.median(window)
            
    return denoised_img
