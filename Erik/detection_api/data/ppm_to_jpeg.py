# Script that will convert all images in GTSDB from PPM to PNG
import glob, os
from PIL import Image

PATH = 'FullIJCNN2013'

all_img_paths = glob.glob(PATH + '/*.ppm')
i = 0
for img_path in all_img_paths:
    head, tail = os.path.split(img_path)
    root, ext = os.path.splitext(tail)
    Image.open(img_path).save('GTSDB/' + root + '.jpeg')
    i = i + 1
    if i % 10 == 0:
        print(str(i) + ' images reformated to jpeg.')

