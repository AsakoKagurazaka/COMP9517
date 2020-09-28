# Copyright 2020 Asako Kagurazaka
import cv2
import numpy as np

# Question 4 Filters
image41 = cv2.imread("imageQ41.png", 0)  # Read image Q4.1
image42 = cv2.imread("imageQ42.png", 0)  # Read image Q4.2

# Show the image and compare
# Q4.1
cv2.imshow("Original Q4.1", image41)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Q4.2
cv2.imshow("Original Q4.2", image42)
cv2.waitKey(0)
cv2.destroyAllWindows()