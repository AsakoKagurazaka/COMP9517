# Copyright 2020 Asako Kagurazaka
import cv2
import numpy as np

# Question 1 Contrast Stretching

image1 = cv2.imread("imageQ1.jpg", 0)  # Read image
minimum = np.min(image1)  # Calculate the minimum value of image1
maximum = np.max(image1)  # Calculate the maximum value of image1

# Create an array to copy the raw data into
x = image1.shape[0]
y = image1.shape[1]
raw = np.zeros((x, y))

# Create a raw data array storing the transformed image
for i in range(x):
    for j in range(y):
        raw[i, j] = ((255-0)*(image1[i, j]-minimum)/(maximum-minimum)+0)

# Make the data CV2-readable by converting it to uint8
image1_t = raw.astype(np.uint8)

# Show the imagesSS
cv2.imshow("Original", image1)
cv2.imshow("Contrast Stretching", image1_t)
cv2.waitKey(0)
cv2.destroyAllWindows()
