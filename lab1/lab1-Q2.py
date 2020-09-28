# Copyright 2020 Asako Kagurazaka
# importing package
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Question2 Histogram

# Read the image
image1 = cv2.imread("imageQ21.jpg")
# Using plt.hist to plot histogram (really don't want to write it myself)
plt.hist(image1.ravel())
plt.show()  # Display the histogram

# Read imageQ22
image2 = cv2.imread("imageQ22.tif")

# Split the image into channels
R, G, B = cv2.split(image2)

# Equalize the image by each channel
output1_R = cv2.equalizeHist(R)  # Red Channel 
output1_G = cv2.equalizeHist(G)  # Green Channel
output1_B = cv2.equalizeHist(B)  # Blue Channel

# Merge the channels
image2_t = cv2.merge((output1_R, output1_G, output1_B))

# Equalization completed, Show the difference

cv2.imshow("Original", image2)
cv2.imshow("Transformed", image2_t)
cv2.waitKey(0)
cv2.destroyAllWindows()