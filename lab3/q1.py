import cv2
import numpy as np
import sys
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth

# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

def mean_shift(filename):
    dim = (480, 320)
    image1 = cv2.imread(filename)
    image1 = cv2.resize(image1, dim)
    image1_r = image1[:, :, 0].flatten()
    image1_g = image1[:, :, 1].flatten()
    image1_b = image1[:, :, 2].flatten()
    image1_list = [image1_b, image1_g, image1_r]
    X = np.stack(image1_list).transpose()
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    image1_labels = np.zeros(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            image1_labels[i][j] = labels[i + j * dim[0]]

    transposed = image1_labels.transpose()
    return transposed

if __name__ == "__main__":
    p1 = plt.imread("imgQ41.jpg")
    p2 = plt.imread("imgQ42.jpg")
    ms1 = mean_shift("imgQ41.jpg")
    ms2 = mean_shift("imgQ42.jpg")
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(p1)
    plt.subplot(222)
    plt.imshow(ms1)
    plt.subplot(223)
    plt.imshow(p2)
    plt.subplot(224)
    plt.imshow(ms2)
    plt.show()