# Copyright 2020 Asako Kagurazaka
import cv2
import numpy as np
# Question 4 Filters
image41 = cv2.imread("imageQ41.png", 1)  # Read image Q4.1
image42 = cv2.imread("imageQ42.png", 1)  # Read image Q4.2

# Q4.1

# Median filters
median3 = cv2.medianBlur(image41, 3)
median5 = cv2.medianBlur(image41, 5)
median7 = cv2.medianBlur(image41, 7)

# Average filters
blur3 = cv2.blur(image41, (3, 3))
blur5 = cv2.blur(image41, (5, 5))
blur7 = cv2.blur(image41, (7, 7))

# Gaussian filters

gauss3 = cv2.GaussianBlur(image41, (3, 3),0)
gauss5 = cv2.GaussianBlur(image41, (5, 5),0)
gauss7 = cv2.GaussianBlur(image41, (7, 7),0)

# Show the image and compare. The title is self-explanatory

# Median, Best result
cv2.imshow("Original Q4.1", image41)
cv2.imshow("Median size = 3", median3) # Best result
cv2.imshow("Median size = 5", median5)
cv2.imshow("Median size = 7", median7)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Average
cv2.imshow("Original Q4.1", image41)
cv2.imshow("Average size = 3", blur3) # Best result
cv2.imshow("Average size = 5", blur5)
cv2.imshow("Average size = 7", blur7)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Gaussian
cv2.imshow("Original Q4.1", image41)
cv2.imshow("Gauss size = 3", gauss3) # Best result
cv2.imshow("Gauss size = 5", gauss5)
cv2.imshow("Gauss size = 7", gauss7)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("The best result is Median filtering & kernel size is 3")

#Q4.2 High pass filter

#A high pass filtered image can also be obtained by subtracting the low pass filtered image from the original image
lpf = cv2.GaussianBlur(image42, (3, 3),0)
hpf = cv2.subtract(image42, lpf)
modified = cv2.add(image42,hpf)
cv2.imshow("Original Q4.2", image42)
cv2.imshow("High Pass Filter", hpf)
cv2.imshow("Filtered", modified)
cv2.waitKey(0)
cv2.destroyAllWindows()