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

def binarize(gray_in, threshold=128):
    return np.where(gray_in > threshold, 255, 0)

def sequential_label(binary_in):
    labelled_img = np.zeros((len(binary_in),len(binary_in[0])))
    equivalences = [] # Equivalence table
    label = 1         # Next label
    for h in range(len(binary_in)):
        top = h == 0
        for w in range(len(binary_in[0])):
            edge = w == 0
            B = 0
            C = 0
            D = 0
            if binary_in[h][w] != 0:
                if not top and not edge:
                    D = labelled_img[h-1][w-1]
                if not top:
                    B = labelled_img[h-1][w]
                if not edge:
                    C = labelled_img[h][w-1]
                if D != 0:
                    labelled_img[h][w] = D
                elif B != 0 and C != 0:
                    labelled_img[h][w] = B
                    if B != C:
                        B_group = -1
                        C_group = -1
                        for g in range(len(equivalences)):
                            if B in equivalences[g]:
                                B_group = g
                            if C in equivalences[g]:
                                C_group = g
                        if B_group == -1 and C_group == -1:
                            equivalences.append([B,C])
                        elif B_group == -1:
                            equivalences[C_group].append(B)
                        elif C_group == -1:
                            equivalences[B_group].append(C)
                        elif B_group != C_group:
                            equivalences[B_group].extend(equivalences[C_group])
                            del equivalences[C_group]
                elif B != 0:
                     labelled_img[h][w] = B
                elif C != 0:
                     labelled_img[h][w] = C
                else:
                    #new label
                    labelled_img[h][w] = label
                    label += 1

    # Equivalence check
    for h in range(len(labelled_img)):
        for w in range(len(labelled_img[0])):
            for g in range(len(equivalences)):
                if labelled_img[h][w] in equivalences[g]:
                    labelled_img[h][w] = g+1

    return labelled_img

def compute_moment(labelled_in):
    moment_dict = {}
    for y in range(len(labelled_in)):
        for x in range(len(labelled_in[y])):
            p = labelled_in[y][x]
            if p != 0:
                if not (p in moment_dict):
                    moment_dict[p] = np.zeros(9)
                moment_dict[p][0] += 1
                moment_dict[p][1] += y
                moment_dict[p][2] += x
                moment_dict[p][3] += y**2
                moment_dict[p][4] += x*y
                moment_dict[p][5] += x**2
    # Central second moments
    for y in range(len(labelled_in)):
        for x in range(len(labelled_in[y])):
            p = labelled_in[y][x]
            if p != 0:
                y_bar = moment_dict[p][1]/moment_dict[p][0]
                x_bar = moment_dict[p][2]/moment_dict[p][0]
                moment_dict[p][6] += (y - y_bar)**2
                moment_dict[p][7] += (x - x_bar)*(y - y_bar)
                moment_dict[p][8] += (x - x_bar)**2
    return moment_dict
