from PIL import Image, ImageDraw, ImageFilter
from random import randint
from blurEdges import blurImage
import csv
import numpy as np
import os
import glob
from countClassFrequency import countFrequency
from matplotlib import pyplot as plt
from EdgeDetection import cannyEdgePIL, cannyEdgeFile

#TODO: Choose nr of images, blur images, define cooridnates and define what to write to annotation if num_sign=0

road_path = 'C:\\Users\\nystr\\PycharmProjects\\Deep-learning-image-recognition\\Lukas\\main_GTSRB\\Augment\\roads\\'
#road_folder = os.listdir(road_path)  # create a folder to iterate through

sign_path = 'C:\\Users\\nystr\\GTSRB\\Final_Test\\Images\\'
#sign_folder = os.listdir(sign_path)  # create a folder to iterate through

aug_path = 'C:\\Users\\nystr\\PycharmProjects\\Deep-learning-image-recognition\\Lukas\\main_GTSRB\\Augment\\aug\\'

files = glob.glob(aug_path+'*.png')
for f in files:
    os.remove(f)
files = glob.glob(aug_path+'\\boxes\\*.png')
for f in files:
    os.remove(f)  #Deletes all old augmented images

#00600.png-10000.png
#00600.png;774;411;815;446;11 in .txt-file, separate rows.
#00600.png;10;10;10;10;25
text_file = open("annotations.txt", "w")
road_size = (1360,800)
max = 610 #00000.png-max.png
number_of_roads = 20
num_classes = 43

filename = 'C:\\users\\nystr\\GTSDB\\gt.txt'
freq_per_class = countFrequency(filename, num_classes)
dist = 1 / freq_per_class / sum(1 / freq_per_class)  # 1/freq_per_class taken from countClassFrequency.py
dist = np.array(dist)
dist /= dist.sum()
# plt.bar(range(len(dist)), dist, 1/1.5, color="blue")
# plt.show()      #Plot distribution

for i in range(600,max+1):
    road_num = randint(1,number_of_roads) #Random road 'road_num.png'
    road = Image.open(road_path+str(road_num)+'.jpg')
    road = road.resize(road_size)
    num_of_signs_in_img = randint(0,6) #number of signs to be pasted into the image
    for j in range(0,num_of_signs_in_img):
        class_nr = np.random.choice(num_classes, p=dist)  #chooses a sign class
        prefix = 'C:\\Users\\nystr\\GTSRB\\Final_Training\\Images\\' + format(class_nr, '05d')
        annotations_path = prefix + '\\' + 'GT-' + format(class_nr, '05d') + '.csv'
        gtFile = open(annotations_path)  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        next(gtReader)  # skip header

        row_num = randint(1, sum(1 for row in gtReader))
        gtFile = open(annotations_path)  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        next(gtReader)  # skip header
        k=1
        for row in gtReader:
            if k==row_num:
                sign = Image.open(prefix + '\\' + row[0])
                x1 = int(row[3])
                y1 = int(row[4])
                x2 = int(row[5])
                y2= int(row[6])
                break;
            k+=1
        x = randint(0, road_size[0]-sign.size[0])
        y = randint(0, road_size[1]-sign.size[1])
        road = blurImage(road,sign,x,y)
        text_file.write(format(i,'05d')+'.png;'+str(x+x1)+';'+str(y+y1)+';'+str(x+x2)+';'+str(y+y2)+';'+str(class_nr)+'\n') #00600.png;774;411;815;446;11 in .txt-file, separate rows.

    #if(num_of_signs_in_img==0):
        #Vad ska skrivas till annotations om ingen skylt?
    road.save(aug_path + format(i, '05d')+'.png')
text_file.close()

#Painting boxes
img_before = None
with open("annotations.txt") as f:
    content = f.readlines()
for i in range(0,len(content)):  #loopa över alla rader i annotations
    semicolon = np.empty(6)
    d=0
    for j in range(0,len(content[i])):
        if(content[i][j] == ";" or content[i][j] == "\n"):
            semicolon[d] = j
            d = d+1
    x1 = int(content[i][int(semicolon[0])+1:int(semicolon[1])])  #koordinater för bounding box
    y1 = int(content[i][int(semicolon[1])+1:int(semicolon[2])])
    x2 = int(content[i][int(semicolon[2])+1:int(semicolon[3])])
    y2 = int(content[i][int(semicolon[3])+1:int(semicolon[4])])

    img_path_file = content[i][:int(semicolon[0])]    #filnamn
    if(img_before==None):
        img = Image.open(aug_path+img_path_file)
    else:
        if(img_path_file!=img_before):
            img.save(aug_path+'\\boxes\\' + img_before)
            img = Image.open(aug_path+img_path_file)
    draw = ImageDraw.Draw(img)
    draw.rectangle(((x1, y1), (x2, y2)),outline='red')
    draw.text((x1, y2), 'Class: '+content[i][int(semicolon[4])+1:int(semicolon[5])],(255,255,255)) #font=ImageFont.truetype("sans-serif.ttf", 16))
    del draw
    #img.show()
    #img.save(aug_path+content[i][:int(semicolon[0])]'.png')
    if(i==len(content)-1):
        img.save(aug_path + '\\boxes\\' + img_path_file)
    img_before = img_path_file

width = 1/1.5
fig = plt.figure()
plt.bar(range(len(freq_per_class)), freq_per_class, width, color="blue",figure=fig)
plt.title('Original dataset')
plt.ylabel('Instances per class')
plt.xlabel('Class ID')
plt.savefig('InstancePerClassOriginal.png')
plt.show()
freq_per_class += countFrequency('annotations.txt',num_classes)

fig2 = plt.figure()
plt.bar(range(len(freq_per_class)), freq_per_class, width, color="blue",figure=fig2)
plt.title('Enlarged dataset')
plt.ylabel('Instances per class')
plt.xlabel('Class ID')
plt.savefig('InstancePerClassEnlarged.png')
plt.show()

