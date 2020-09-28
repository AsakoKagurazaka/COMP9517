# Copyright 2020 Asako Kagurazaka
# importing package
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Question3 Edges
# Code transcribed from https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[0]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output
# Pre-defined matrices and constants
F_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
F_y = [[-1,-2,-1], [0, 0, 0], [1, 2, 1]]

scale = 1
delta = 0
ddepth = cv2.CV_16S

# Read the image
image3 = cv2.cvtColor(cv2.imread("imageQ3.jpg"),cv2.COLOR_BGR2GRAY)

# Do the convolution
image3_cx = convolve2D(image3, F_x)
image3_cy = convolve2D(image3, F_y)

# Sobel method
grad_x = cv2.Sobel(image3, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(image3, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
# Absolute value
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

# Show the images
cv2.imshow("Original", image3)
cv2.imshow("Convolution X axis", image3_cx)
cv2.imshow("Convolution Y axis", image3_cy)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Due to hardware limitations, the image displaying sessions are split in two, press any key to go to the next session
cv2.imshow("Convolution X axis", abs_grad_x)
cv2.imshow("Convolution Y axis", abs_grad_y)
cv2.waitKey(0)
cv2.destroyAllWindows()