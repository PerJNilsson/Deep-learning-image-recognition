from PIL import Image, ImageDraw, ImageFilter
from random import randint
import csv
from EdgeDetection import cannyEdgePIL, cannyEdgeFile
import numpy as np
import os
from blurEdges import blurImage
import glob

x = [1, 2, 3, 4]
print(x[1:])