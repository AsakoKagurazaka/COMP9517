import os
from random import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


# misc
def is_border(location, w, h):
    if location[0] == 0 or location[0] == w or location[1] == h or location[1] == 0:
        return 1
    return 0


def bin_image_gen(iteration, t, original_image):
    bin_image = np.zeros(original_image.shape)
    w, h = bin_image.shape
    for i in range(w):
        for j in range(h):
            if original_image[i, j] > t:
                bin_image[i, j] = 0
            else:
                bin_image[i, j] = 255
    cv2.imwrite("OUTPUT/"+str(iteration)+".png", bin_image)


# calculate a threshold value
def calc_threshold(file):
    sigma = 0.01
    iter_count = []
    iter_value = []
    t = 64 + random() * 128
    print("initial random value is " + str(t))
    image = cv2.imread(file, 0)
    w, h = image.shape
    u0 = 0.0
    u0_count = 0
    u1 = 0.0
    u1_count = 0
    t1 = 0.0
    iteration = 0
    # debug
    while t - t1 > sigma or t - t1 < -sigma:
        iteration += 1
        t1 = t
        for i in range(w):
            for j in range(h):
                if image[i, j] < t:
                    u0 += image[i, j]
                    u0_count += 1
                else:
                    u1 += image[i, j]
                    u1_count += 1
        u0 /= u0_count
        u1 /= u1_count
        u0_count = 0
        u1_count = 0
        t = (u0+u1)/2
        print("iteration count " + str(iteration))
        print("u0 = " + str(u0) + ", u1 = " + str(u1) + ", t= " + str(t))
        iter_count.append(iteration)
        iter_value.append(t)
        bin_image_gen(iteration, t, image)
    plt.plot(iter_count, iter_value)
    plt.xlabel('iteration')
    plt.ylabel('t')
    plt.show()
    return t


if __name__ == "__main__":
    img = "rice_img6.png"
    print("t = " + str(calc_threshold(img)))
