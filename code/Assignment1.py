# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 23:05:46 2020

@author: 
"""

# ......IMPORT .........
import argparse
import os
from collections import Counter
from random import random
import cv2
import matplotlib.pyplot as plt
import numpy as np


# misc
# a Union-find Array implementation
# Reference code is https://github.com/spwhitt/cclabel/
class UFarray:
    def __init__(self):
        # Array which holds label -> set equivalences
        self.labels = []

        # Name of the next label, when one is created
        self.label = 0

    def make_label(self):
        r = self.label
        self.label += 1
        self.labels.append(r)
        return r

    def set_root(self, i, root):
        while self.labels[i] < i:
            j = self.labels[i]
            self.labels[i] = root
            i = j
        self.labels[i] = root

    def find_root(self, i):
        while self.labels[i] < i:
            i = self.labels[i]
        return i

    def find(self, i):
        root = self.find_root(i)
        self.set_root(i, root)
        return root

    def union(self, i, j):
        if i != j:
            root = self.find_root(i)
            root_j = self.find_root(j)
            if root > root_j:
                root = root_j
            self.set_root(j, root)
            self.set_root(i, root)

    def flatten(self):
        for i in range(1, len(self.labels)):
            self.labels[i] = self.labels[self.labels[i]]


def not_background(bin_image, i, j):
    if bin_image[i, j] == 0:
        return False
    return True


def bin_image_gen(t, original_image):
    bin_image = np.zeros(original_image.shape)
    w, h = bin_image.shape
    for i in range(w):
        for j in range(h):
            if original_image[i, j] > t:
                bin_image[i, j] = 0
            else:
                bin_image[i, j] = 255
    return bin_image


# Median filter algorithm, with padding
def median_filter(origin):
    w, h = origin.shape
    padded = np.full((w + 4, h + 4), 255)
    for y in range(w):
        for x in range(h):
            padded[y + 2, x + 2] = origin[y, x]

    for y in range(2, w + 2):
        for x in range(2, h + 2):
            members = []
            for i in range(-2, 3):
                for j in range(-2, 3):
                    members.append(padded[y + i, x + j])
            members.sort()
            padded[y, x] = members[12]

    result = padded[2:w + 2, 2:h + 2]

    return result


# Calculate a threshold value
def iso_data_thresholding(file):
    sigma = 0.01
    iter_count = []
    iter_value = []
    t = 96 + random() * 64
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
        t = (u0 + u1) / 2
        iter_count.append(iteration)
        iter_value.append(t)
    plt.plot(iter_count, iter_value)
    plt.title('Convergence curve')
    plt.xlabel('iteration')
    plt.ylabel('threshold')
    plt.show()
    return t, bin_image_gen(t, image)


# Two-pass algorithm
# The instructor approved the use of an external paper, will be within section 5 of the report:
# https://piazza.com/class/kemg8hc3toq3td?cid=164,
# Report located at https://sdm.lbl.gov/~kewu/ps/paa-final.pdf
# Reference code is https://github.com/spwhitt/cclabel/
# Modifications are done to fulfill the assignment's requirement.
def two_pass(data):
    w, h = data.shape
    uf = UFarray()
    label = {}

    for j in range(h):
        for i in range(w):
            # we only consider P1 P2 P3 and P8 since P4-P7 will be 0 in the first pass
            if data[i, j] != 0:
                pass
            # consider P2
            elif j > 0 and data[i, j - 1] == 0:
                label[i, j] = label[(i, j - 1)]
            # consider P3
            elif i + 1 < w and j > 0 and data[i + 1, j - 1] == 0:
                c = label[(i + 1, j - 1)]
                label[i, j] = c
                # if P1 and P3 are both not in background, connect
                if i > 0 and data[i - 1, j - 1] == 0:
                    a = label[(i - 1, j - 1)]
                    uf.union(c, a)
                # if P8 and P3 are both not in background, connect
                elif i > 0 and data[i - 1, j] == 0:
                    d = label[(i - 1, j)]
                    uf.union(c, d)
            # consider P1
            elif i > 0 and j > 0 and data[i - 1, j - 1] == 0:
                label[i, j] = label[(i - 1, j - 1)]
            # consider P8
            elif i > 0 and data[i - 1, j] == 0:
                label[i, j] = label[(i - 1, j)]
            else:
                label[i, j] = uf.make_label()
    # connect the components
    uf.flatten()

    # second pass
    for (i, j) in label:
        component = uf.find(label[(i, j)])
        label[(i, j)] = component

    label_length = []
    for i in label.values():
        if i not in label_length:
            label_length.append(i)

    ct = len(label_length)

    return ct, label


# task 3
def remove_bad(image, label, bad_threshold):
    counter = Counter(label.values())
    bad_rice = []
    for x in counter:
        if counter[x] < bad_threshold and x not in bad_rice:
            bad_rice.append(x)

    for x, y in label.items():
        for i in bad_rice:
            if y == i:
                image[x[0], x[1]] = 255

    return len(bad_rice), image


# main function
my_parser = argparse.ArgumentParser()
my_parser.add_argument('-o', '--OP_folder', type=str, help='Output folder name', default='OUTPUT')
my_parser.add_argument('-m', '--min_area', type=int, action='store', required=True,
                       help='Minimum pixel area to be occupied, to be considered a whole rice kernel')
my_parser.add_argument('-f', '--input_filename', type=str, action='store', required=True, help='Filename of image ')
# Execute parse_args()
args = my_parser.parse_args()

# someone deleted my file, this one is written in a hurry so no failsafe. Sorry!

output = args.OP_folder
minimum = args.min_area
filename = args.input_filename
if not os.path.isdir(output):
    os.mkdir(output)

print("Running...")
threshold, image1 = iso_data_thresholding(filename)

plt.title("Threshold value = %.2f" % threshold)
plt.axis("off")
plt.imshow(image1, cmap='gray', vmin=0, vmax=255)
plt.savefig(output + "/" + filename + "_Task1.png")
print("Task 1 completed, running task 2...")
filtered = median_filter(image1)

count, labels = two_pass(filtered)

plt.title("Rice count = %d" % count)
plt.axis("off")
plt.imshow(filtered, cmap='gray', vmin=0, vmax=255)
plt.savefig(output + "/" + filename + "_Task2.png")
print("Task 2 completed, running task 3...")

bad, bad_removed = remove_bad(filtered, labels, minimum)

plt.title("Damaged count = %.5f %%" % (bad/count*100.0))
plt.axis("off")
plt.imshow(bad_removed, cmap='gray', vmin=0, vmax=255)
plt.savefig(output + "/" + filename + "_Task3.png")

print("Finished!")
