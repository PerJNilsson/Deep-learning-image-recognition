from PIL import Image, ImageDraw, ImageFilter
from random import randint
import csv
from EdgeDetection import cannyEdgePIL, cannyEdgeFile
import numpy as np
import os
import glob


def blurImage(road, sign, x, y):
    road.paste(sign, (x, y))
    #road.show()
    num_of_it = 5
    for bl in range(num_of_it):
        frac = (bl + 1) / 100
        marginx = round(sign.size[0] * frac)
        marginy = round(sign.size[1] * frac)
        left_side = road.crop((x - marginx, y - marginy, x + marginx, y + sign.size[1] + marginy))  # Left edge of transition
        right_side = road.crop((x + sign.size[0] - marginx, y - marginy, x + sign.size[0] + marginx,y + sign.size[1] + marginy))  # Left edge of transition
        top_side = road.crop((x - marginx, y - marginy, x + sign.size[0] + marginx, y + marginy))  # Left edge of transition
        bottom_side = road.crop((x - marginx, y + sign.size[1] - marginy, x + sign.size[0] + marginx,y + sign.size[1] + marginy))  # Left edge of transition

        left_side = left_side.filter(ImageFilter.GaussianBlur)
        right_side = right_side.filter(ImageFilter.GaussianBlur)
        top_side = top_side.filter(ImageFilter.GaussianBlur)
        bottom_side = bottom_side.filter(ImageFilter.GaussianBlur)

        road.paste(left_side, (x - marginx, y - marginy))
        road.paste(right_side, (x + sign.size[0] - marginx, y - marginy))
        road.paste(top_side, (x - marginx, y - marginy))
        road.paste(bottom_side, (x - marginx, y + sign.size[1] - marginy))
        #if bl % 5 == 0:
        #    road.show()
    return road

