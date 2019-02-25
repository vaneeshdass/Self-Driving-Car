import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in an image
img = cv2.imread('sobel-example.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate the derivative in the xx direction (the 1, 0 at the end denotes xx direction)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
#Calculate the derivative in the yy direction (the 0, 1 at the end denotes yy direction)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

#this absolute fns turn all negative values into positive
abs_sobelx = np.absolute(sobelx)

scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
plt.imshow(sxbinary, cmap='gray')
plt.show()

