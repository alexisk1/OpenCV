import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 15)


img = mpimg.imread("./test1.jpg")
gray = gaussian_blur(img, 9)
gray = grayscale(gray)
cny =canny(img,30,100)

plt.imshow(img)
plt.show()
contours, _ = cv2.findContours(cny.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key = cv2.contourArea, reverse = True)  #to limit, add [:20] to the end of the array

for i in contours:
    cv2.drawContours(img, i, -1, (0,255,0), 3)
  
    plt.imshow(img)
    plt.show()

