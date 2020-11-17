import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
import csv
import re

from joblib import load, dump

def load_plant_names(): 
    """
    Read files ending with '_rgb.png' and files ending with '_fg.png'
    Return the images as tuple of numpy arrays
    """
    path_names = [
        'Plant_Phenotyping_Datasets/Tray/Ara2012/',
        'Plant_Phenotyping_Datasets/Tray/Ara2013-Canon/'
    ]
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
    return plant_names

X, Y_imgs = load('numpy_images.joblib')
img_dim = Y_imgs[0].shape
plant_names = load_plant_names()
X_train, X_test, Y_train, Y_test, plant_names_train, _ = train_test_split(
    X, Y_imgs, plant_names, train_size=10, random_state=3
)

gmm = load('model.joblib')
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

# copy = preds[1]
preds = list(map(post_processing, preds))
# fig, axs = plt.subplots(1, 3)
# axs[0].imshow(preds[1])
# axs[1].imshow(Y_test[1], cmap='gray')
# axs[2].imshow(copy)
# plt.show()

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