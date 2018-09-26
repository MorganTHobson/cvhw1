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
