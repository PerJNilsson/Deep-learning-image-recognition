from PIL import Image, ImageDraw, ImageFilter
from random import randint
import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt

def cannyEdgeFile(filename):
    img = cv2.imread(filename, 0)
    edges = cv2.Canny(img, 100, 200)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def cannyEdgePIL(img1):
    img = np.array(img1)
    # Convert RGB to BGR
    img = img[:, :, ::-1].copy()
    edges = cv2.Canny(img, 100, 200)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    plt.close()
