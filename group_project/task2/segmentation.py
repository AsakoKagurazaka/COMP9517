import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

# dataloading modules
from os import listdir
from os.path import isfile, join
import re
import random
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from joblib import dump

# clustering modules
from sklearn.mixture import GaussianMixture

import data_loader


def main():
    X, Y, plant_names = data_loader.main()
    X_train, X_test, Y_train, Y_test, plant_names_train, _ = train_test_split(
        X, Y, plant_names, train_size=20, random_state=3
    )
    print("Training on {} images and test on {} images"
        .format(len(X_train), len(X_test))
    )
    gmm = GaussianMixture(n_components=3, random_state=1, warm_start=True)
    print("Training...")
    for item, target in zip(X_train, Y_train):
        gmm.fit(item.reshape(-1, 1))
    print("Predicting...")
    preds = []
    for item, target in zip(X_test, Y_test):
        pred_image = gmm.predict(item.reshape(-1, 1))
        pred = pred_image.reshape(img_dim)
        preds.append(pred)
        
    print('Post processing...')
    def post_processing(img):
        # remove third class in Gaussian Mixture
        img = np.where(img==2, 0, img)
        # convert to binary
        img = np.where(img==1, 255, img)
        img = np.uint8(img)
        # remove salt and pepper noise
        img = cv2.medianBlur(img, 7)
        return img

    preds = list(map(post_processing, preds))

    def dice_coefficient(img1, img2):
        img1 = np.asarray(img1).astype(np.bool)
        img2 = np.asarray(img2).astype(np.bool)
        if img1.shape != img2.shape:
            raise ValueError("Shape mismatch: img1 and img2 must have the same shape.")

        # Compute Dice coefficient
        intersection = np.logical_and(img1, img2)
        return 2. * intersection.sum() / (img1.sum() + img2.sum())

    def intersection_over_union(img1, img2):
        intersection = cv2.bitwise_and(img1, img2)
        union = cv2.bitwise_or(img1, img2)
        return np.count_nonzero(intersection)/np.count_nonzero(union)
        

    dice_score = 0
    iou = 0
    for pred, target in zip(preds, Y_test):
        dice_score += dice_coefficient(pred, target)
        iou += intersection_over_union(pred, target)
    dice_score /= len(Y_test)
    iou /= len(Y_test)
    print("Dice: ", dice_score)
    print("IOU: ", iou)

if __name__ == '__main__':
    main()
