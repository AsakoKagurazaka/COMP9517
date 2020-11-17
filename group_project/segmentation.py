import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

# dataloading modules
from os import listdir
from os.path import isfile, join
import csv
import re
import random
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from joblib import dump

# clustering modules
from sklearn.mixture import GaussianMixture


def load_data() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Read files ending with '_rgb.png' and files ending with '_fg.png'
    Return the images as tuple of numpy arrays
    """
    path_names = [
        'Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Tray/Ara2012/',
        'Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Tray/Ara2013-Canon/'
    ]

    X = []
    Y = []
    plant_names = []

    for path_name in path_names:
        files = [f for f in listdir(path_name) if isfile(join(path_name, f))]
        for file in files:
            if not file.endswith('rgb.png'):
                continue

            # extract the part before _rgb.png
            file_pattern = re.match(r'(.*)_rgb\.png$', file)
            plant_name = file_pattern.group(1)
            plant_names.append(path_name + plant_name)
            segmented_img_file = plant_name + '_fg.png'

            # read image
            img = cv2.imread(path_name + file)
            X.append(img)
            segmented_img = cv2.imread(path_name + segmented_img_file, 0)
            Y.append(segmented_img)

    return (X, Y, plant_names)


def diff_of_gaussian(img_l: np.ndarray):
    """
    :param img_l: L channel of L*a*b encoding
    """
    low_sigma = cv2.GaussianBlur(img_l, (3, 3), sigmaX=1, sigmaY=1)
    high_sigma = cv2.GaussianBlur(img_l, (5, 5), sigmaX=4, sigmaY=4)

    # Calculate the DoG by subtracting
    dog = low_sigma - high_sigma
    return dog


def pillbox_filter(img_a):
    """
    :param img_a: a channel of L*a*b encoding
    """
    # estimation of pillbox filter
    # from https://www2.cs.duke.edu/courses/spring19/compsci527/notes/convolution-filtering.pdf
    kernel = np.array([
        [0.0511, 0.0511, 0.0511],
        [0.0511, 0.0511, 0.0511],
        [0.0511, 0.0511, 0.0511],
    ])
    return cv2.filter2D(img_a, -1, kernel)


def feature(img_bgr: np.ndarray):
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    transposed_img = img_lab.transpose(2, 0, 1)

    dog = diff_of_gaussian(transposed_img[0])
    p = pillbox_filter(transposed_img[1])

    normalise = lambda x: math.exp(-1/50*abs(x))
    pointwise_norm = np.vectorize(normalise)
    tfb = pointwise_norm(dog+p)
    return tfb


def preprocessing(X):
    features = list(map(feature, X))
    return features


def main():
    print("loading images")
    X_imgs, Y_imgs, plant_names = load_data()
    # scaling the images
    scale_percentage = 1
    X_imgs = list(map(lambda img: cv2.resize(
        img, (0, 0), fx=scale_percentage, fy=scale_percentage), X_imgs))
    Y_imgs = list(map(lambda img: cv2.resize(
        img, (0, 0), fx=scale_percentage, fy=scale_percentage), Y_imgs))

    img_dim = Y_imgs[0].shape

    print("preprocessing")
    X = preprocessing(X_imgs)
    X_train, X_test, Y_train, Y_test, plant_names_train, _ = train_test_split(
        X, Y_imgs, plant_names, train_size=10, random_state=3
    )
    # persisting train-test data for quick experiment
    dump(X_test, 'test_data.joblib')
    print("Training on {} images and test on {} images"
        .format(len(X_train), len(X_test))
    )

    gmm = GaussianMixture(n_components=3, random_state=1, warm_start=True)
    print("Training...")
    for item, target in zip(X_train, Y_train):
        gmm.fit(item.reshape(-1, 1))
    # persist model
    dump(gmm, 'model.joblib')
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