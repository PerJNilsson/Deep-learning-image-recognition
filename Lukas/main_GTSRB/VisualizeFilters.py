from __future__ import print_function
from keras.models import load_model
from keras.utils import plot_model
import numpy as np
from PIL import Image

#####################################################
#####################################################
# Below are two scripts. The first is to print all filters in one image. The second in separate images
# The filters are from arch_2 conv_layer1

model = load_model('C:\\Users\\nystr\\Chalmers Teknologkonsulter AB\\Bird Classification - Images\\arch2_model')
print('loaded')

filters_conv1 = model.layers[1].get_weights()[0]
biases_conv1 = model.layers[1].get_weights()[1]
filtersize = filters_conv1.shape[0]
num_filters = filters_conv1.shape[3]
margin = 1
final_size_img = 128
all_filters_img = Image.new('1',(final_size_img*int(np.sqrt(num_filters)),final_size_img*int(np.sqrt(num_filters))))   #Assumes x^2 nr of filters
#RGB_: all_filters_img = Image.new('RGB',(final_size_img*int(np.sqrt(num_filters)),final_size_img*int(np.sqrt(num_filters))))   #Assumes x^2 nr of filters
for j in range(0,num_filters):
    filter_gray = (filters_conv1[:, :, 0, j] + filters_conv1[:,:,1,j] + filters_conv1[:,:,2,j])*1/3
    temp_matrix = np.zeros((filtersize + margin, filtersize + margin))
    temp_matrix[:filtersize, :filtersize] = filter_gray  # Pastes our 7*7 BW_matrix into the upper left corner of an empty 8*8 matrix
    imgBW = Image.fromarray(temp_matrix, mode='1')
    imgBW = imgBW.resize((final_size_img, final_size_img))
    #imgBW.show()
    #temp_matrix = np.zeros((filtersize+marg,filtersize+marg,3))
    #temp_matrix[:filtersize,:filtersize,:] = filters_conv1[:,:,:,j]   #Pastes our 7*7*3 into the upper left corner of an empty 8*8*3 matrix
    #imgRGB = Image.fromarray(temp_matrix,mode='RGB')
    #imgRGB = imgRGB.resize((128,128))
    x1 = final_size_img*(j%8)  #x and y below assume 64 filters
    y1 = final_size_img*int(j/8)
    x2 = x1 + final_size_img
    y2 = y1 + final_size_img
    all_filters_img.paste(imgBW,(x1,y1,x2,y2))
all_filters_img.show()

#plot_model(model, to_file='arch2_model.png')
#for i in range(0,len(model.layers)):
#weights = model.layers[1].get_weights()
#print(weights)

#########################################################
#########################################################
#Below is the script to print all images in BW,Gray,RGB


#model = load_model('C:\\Users\\nystr\\Chalmers Teknologkonsulter AB\\Bird Classification - Images\\arch2_model')
#print('loaded')
#filters_conv1 = model.layers[1].get_weights()[0]
#biases_conv1 = model.layers[1].get_weights()[1]
#num_filters = filters_conv1.shape[3]
#for j in range(0,num_filters):
    #filter_gray = (filters_conv1[:, :, 0, j] + filters_conv1[:,:,1,j] + filters_conv1[:,:,2,j])*1/3
    #imgBW = Image.fromarray(filter_gray, mode='1')
    #imgBW = imgBW.resize((256,256))
    #imgBW.show()  #Black and white
    #imgG = Image.fromarray(filter_gray,mode='L')
    #imgG = imgG.resize((256,256))
    #imgG.show()  #Gray (8-bit per pixel)
    #imgRGB = Image.fromarray(filters_conv1[:,:,:,j],mode='RGB')
    #imgRGB= imgRGB.resize((256,256))
    #imgRGB.show() #RGB