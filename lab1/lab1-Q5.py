# Copyright 2020 Asako Kagurazaka
import cv2
import numpy as np

# Question 5 Restore image

# Using OpenCV functions as per https://piazza.com/class/kemg8hc3toq3td?cid=79 (The lecturer said so)
image1 = cv2.imread("imageQ5.png", 0)  # Read image
cv2.imshow("Original Image", image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply a low pass filter in Q4
lpfed = cv2.medianBlur(image1, 3)
cv2.imshow("LPFed", lpfed)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
# Increase the contrast of the image
minimum = np.min(lpfed)  # Calculate the minimum value of image1
maximum = np.max(lpfed)  # Calculate the maximum value of image1
x = lpfed.shape[0]
y = lpfed.shape[1]
raw = np.zeros((x, y))
for i in range(x):
    for j in range(y):
        raw[i, j] = ((255-0)*(lpfed[i, j]-minimum)/(maximum-minimum)+0)
cont = raw.astype(np.uint8)
cv2.imshow("Increased Contrast", cont)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Perform a High Pass Filter
lpf = cv2.GaussianBlur(cont, (5, 5),0)
hpf = cv2.subtract(cont, lpf)
hpfed = cv2.add(cont,hpf)

cv2.imshow("High Pass Filter", hpfed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Perform histogram equalization
hist_eq = cv2.equalizeHist(hpfed)
cv2.imshow("Histogram Equalization", hist_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Show the result
result = hist_eq
cv2.imshow("Restored Image", result)

# Save the result to file. The code is ran once before submitting.
cv2.imwrite("result.png", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
