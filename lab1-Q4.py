# Copyright 2020 Asako Kagurazaka
import cv2
import numpy as np
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