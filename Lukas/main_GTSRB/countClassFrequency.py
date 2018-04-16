from PIL import Image, ImageDraw
import csv
import numpy as np
from matplotlib import pyplot as plt

def countFrequency(filename, num_classes):
    #filename = 'C:\\users\\nystr\\GTSDB\\gt.txt'
    freq_per_class = np.zeros(num_classes)  #increase index "classID" each time a class is used
    with open(filename) as f:
        content = f.readlines()
    for i in range(len(content)):  #loopa över alla rader i annotations
        semicolon = np.empty(6)
        d=0
        for j in range(len(content[i])):
            if(content[i][j] == ";" or content[i][j] == "\n"):
                semicolon[d] = j
                d = d+1
        classID = int(content[i][int(semicolon[4])+1:int(semicolon[5])])  #classID for the current image
        freq_per_class[classID] +=1

    #width = 1/1.5
    #print(freq_per_class)
    #print(len(freq_per_class))
    #fig = plt.figure()
   # plt.bar(range(len(freq_per_class)), freq_per_class, width, color="blue")
   # plt.ylabel('Instances per class')
    #plt.xlabel('Class ID')
    #plt.savefig('InstancePerClass.png')
    #plt.show()

    #dist = 1/freq_per_class/sum(1/freq_per_class)
    #dist[0] += 1-sum(dist)
    #plt.bar(range(len(freq_per_class)), dist, width, color="blue")
    #plt.show()
    return freq_per_class

