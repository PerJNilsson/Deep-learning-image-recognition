from __future__ import print_function
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Model
import keras
from matplotlib import colorbar
from sklearn import manifold
from sklearn import decomposition
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


def NNPixel(x_test,y_test, index,number_of_neighbors, num_classes):
    choosen_img = x_test[index]
    #plt.imshow(choosen_img)
    #plt.show()
    loneerror, ltwoerror = np.zeros(len(x_test)),np.zeros(len(x_test))   # L1 and L2 error vector
    for i in range(len(x_test)):
        temp = x_test[i]-choosen_img
        abs_temp = np.abs(temp)
        pow_temp = np.square(temp)
        loneerror[i] = np.sum(abs_temp)  # L1 error vector
        ltwoerror[i] = np.sum(pow_temp)  # L2 error vector

    index_min_lone, index_min_ltwo = np.zeros(number_of_neighbors).astype(int), np.zeros(number_of_neighbors).astype(int)
    for k in range(number_of_neighbors):
        index_min_lone[k] = loneerror.argmin()
        index_min_ltwo[k] = ltwoerror.argmin()
        loneerror[index_min_lone[k]] = np.power(10,30)
        ltwoerror[index_min_lone[k]] = np.power(10,30)

    #print(index_min_lone)
    #print(index_min_ltwo)
    #for p in range(0,number_of_neighbors):
        #plt.imshow(x_test[index_min_lone[p]])
        #plt.show()
        #plt.imshow(x_test[index_min_ltwo[p]])
        #plt.show()

    NNlabelsLone = np.zeros(number_of_neighbors).astype(int)
    NNlabelsLtwo = np.zeros(number_of_neighbors).astype(int)

    for nn in range(number_of_neighbors):
        NNlabelsLone[nn] = np.argmax(y_test[index_min_lone[nn]])
        NNlabelsLtwo[nn] = np.argmax(y_test[index_min_ltwo[nn]])
    print('L1 prediction, with '+str(number_of_neighbors)+' NN: ' + str(KNNprediction(NNlabelsLone, num_classes)))
    print('L1 class prediction, with '+str(number_of_neighbors)+' NN: ' + str(np.argmax(KNNprediction(NNlabelsLone, num_classes))))
    #print('L2 prediction, with '+str(number_of_neighbors)+' NN: ' + str(KNNprediction(NNlabelsLtwo, num_classes)))
    #print('L2 class prediction, with '+str(number_of_neighbors)+' NN: ' + str(np.argmax(KNNprediction(NNlabelsLtwo, num_classes))))

    print('True label: ' + str(np.argmax(y_test[index])))

def NNFeature(model, x_test, y_test, index, img_x, img_y, layer_name, number_of_neighbors, num_classes):
    # Gives us the NN-images and can show these
    # Gives us a K-NN prediction class
    # Gives us a t-sne reduced feature space

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = np.asarray([intermediate_layer_model.predict(i.reshape(1, img_x, img_y, 3)) for i in x_test])
    loneerror, ltwoerror = np.zeros(len(x_test)),np.zeros(len(x_test))   # L1 and L2 error vector
    for i in range(len(x_test)):
        loneerror[i] = np.sum(np.abs(intermediate_output[index]-intermediate_output[i]))
        ltwoerror[i] = np.sum(np.square(intermediate_output[index]-intermediate_output[i]))

    #print(loneerror)
    #print(ltwoerror)

    index_min_lone, index_min_ltwo = np.zeros(number_of_neighbors).astype(int), np.zeros(number_of_neighbors).astype(int)
    for k in range(number_of_neighbors):
        index_min_lone[k] = loneerror.argmin()
        index_min_ltwo[k] = ltwoerror.argmin()
        loneerror[index_min_lone[k]] = np.power(10,30)
        ltwoerror[index_min_lone[k]] = np.power(10,30)
        #plt.imshow(x_test[index_min_lone[k]])
        #plt.show()
        #plt.imshow(x_test[index_min_ltwo[k]])
        #plt.show()

    #print(index_min_lone)
    #print(index_min_ltwo)

    NNlabelsLone = np.zeros(number_of_neighbors).astype(int)
    NNlabelsLtwo = np.zeros(number_of_neighbors).astype(int)

    for nn in range(number_of_neighbors):
        NNlabelsLone[nn] = np.argmax(y_test[index_min_lone[nn]])
        NNlabelsLtwo[nn] = np.argmax(y_test[index_min_ltwo[nn]])
    print('L1 prediction, with '+str(number_of_neighbors)+' NN: ' + str(KNNprediction(NNlabelsLone, num_classes)))
    print('L1 class prediction, with '+str(number_of_neighbors)+' NN: ' + str(np.argmax(KNNprediction(NNlabelsLone, num_classes))))
    #print('L2 prediction, with '+str(number_of_neighbors)+' NN: ' + str(KNNprediction(NNlabelsLtwo, num_classes)))
    #print('L2 class prediction, with '+str(number_of_neighbors)+' NN: ' + str(np.argmax(KNNprediction(NNlabelsLtwo, num_classes))))

    print('True label: ' + str(np.argmax(y_test[index])))

    ################### Perform t-sne
    tsne(intermediate_output, y_test)
    #tsne3D(intermediate_output, y_test)

def KNNprediction(NNlabels, num_classes):
    return np.bincount(NNlabels, minlength=num_classes) / len(NNlabels)

def tsne(intermediate_output, y_test):
    N = len(y_test[0])
    intermediate_output = intermediate_output[:, 0, :]
    #First do PCA to reduce complexity, to dim=50
    pca = decomposition.PCA(n_components=50)
    pca_result = pca.fit_transform(intermediate_output)

    tsne = manifold.TSNE(n_components=2, verbose=0, perplexity=40, n_iter=500)
    #tsne_results = tsne.fit_transform(intermediate_output,y_test)    #Withoout PCA: Converts the feature-space vector to a 2D-vector (len(intermediate_output), 2)
    tsne_results = tsne.fit_transform(pca_result,y_test)    #With PCA: Converts the feature-space vector to a 2D-vector (len(intermediate_output), 2)

    #for i in range(len(tsne_results)):
    #    y_temp = np.argmax(y_test[i])
    #    print('Class nr: '+ str(y_temp)+ ', which corresponds to the 2D-vector: '+str(tsne_results[i]))
    x = tsne_results[:,0]
    y = tsne_results[:,1]
    fig = plt.figure(figsize=(8, 8))

    y_from_cath = np.zeros(len(y_test))
    for i in range(len(y_test)):
        y_from_cath[i] = np.argmax(y_test[i])

    ################ To define manually which color belongs to each class
    #labl = [0, 1, 1, 0, 0]
    #color = ['red' if l == 0 else 'green' for l in labl]
    #plt.scatter(x, y, color=color)
    #############Below: Uses Jet-colors, pre-defiend

    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0, N, N + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    scat = plt.scatter(x, y, c=y_from_cath,cmap=cmap, norm=norm)
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Colour as a function of class')
    plt.title('T-Sne visualization of the feature space')
    plt.savefig('tsne.png')
    plt.show()

def activationMap(model, x_test, img_x, img_y, layer_name, num_NN):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = np.asarray([intermediate_layer_model.predict(i.reshape(1, img_x, img_y, 3)) for i in x_test])
    intermediate_output = intermediate_output[:,0,:,:,:]
    num_of_filters = intermediate_output.shape[3]
    for img_nr in range(1,len(x_test)):
        for filter_nr in range(num_of_filters):
            #plt.imshow(x_test[img_nr])
            #plt.show()
            plt.imshow(intermediate_output[img_nr,:,:,filter_nr])
            plt.colorbar()
            plt.title('Activationmap of image x_test['+str(img_nr)+'], convolved with filter '+ str(filter_nr))
            plt.show()

def occlusion(model, image, label, filter_frac):
    # Image is a (img_x, img_y, 3) numpy array

    true_class = np.argmax(label)
    true_pred_full = model.predict(image.reshape(1, int(image.shape[0]), int(image.shape[1]), int(image.shape[2])), verbose=0)
    predicted_class = np.argmax(true_pred_full)
    if true_class!=predicted_class:
        print('Fel prediction!')
        return
    true_pred_acc = true_pred_full[0, true_class]
    filter_img = np.zeros((round(image.shape[0]*filter_frac), round(image.shape[1]*filter_frac), 3))  #frac=0.15 -> 5x5, 0.3 -> 10x10
    temp_image = np.zeros(image.shape)
    predictions = np.ones((4, int(image.shape[0]), int(image.shape[1])))*true_pred_acc    #First index: 3=RGB, 0=R, 1=G, 2=B (not sure about colorchannels, in order)
    final_filter_images = np.zeros((4, int(image.shape[0]), int(image.shape[1]), int(image.shape[2])))
    for posx in range(image.shape[0]-filter_img.shape[0]+1):
        for posy in range(image.shape[1]-filter_img.shape[1]+1):
            for ch in range(4):
                temp_image[:, :, :] = image[:, :, :]
                if ch==3:
                    temp_image[posy:filter_img.shape[0]+posy, posx:filter_img.shape[1]+posx, :] = filter_img[:, :, :]
                else:
                    temp_image[posy:filter_img.shape[0]+posy, posx:filter_img.shape[1]+posx, ch] = filter_img[:, :, ch]
                y_pred_full = model.predict(temp_image.reshape(1, int(temp_image.shape[0]), int(temp_image.shape[1]), int(temp_image.shape[2])), verbose=0)
                pred_acc = y_pred_full[0, true_class]
                predictions[ch, posy+int(filter_img.shape[1]/2), posx+int(filter_img.shape[0]/2)] = pred_acc
                if posx == posy == image.shape[0]-filter_img.shape[0]:
                    final_filter_images[ch,:,:,:] = temp_image[:,:,:]

    lowest_pred_scores = np.asarray([np.min(row) for row in predictions])  #For each occlusion test (3 channel, 1 whole)
    #highest_pred_scores = np.asarray([np.max(row) for row in predictions])

    flattened_index_lowest_pred_scores = np.asarray([np.argmin(row) for row in predictions])  #For each occlusion test (3 channel, 1 whole)
    x_index_lowest_pred_scores = flattened_index_lowest_pred_scores%image.shape[0]
    y_index_lowest_pred_scores = flattened_index_lowest_pred_scores/image.shape[0]
    y_index_lowest_pred_scores = y_index_lowest_pred_scores.astype('int8')

    fig = plt.figure(figsize=(16, 16))

    for ch in range(4):
        posx = x_index_lowest_pred_scores[ch]
        posy = y_index_lowest_pred_scores[ch]
        #print('Koordinat: ('+str(posx)+','+str(posy)+')')
        if posy < int(filter_img.shape[1]/2) or posx < int(filter_img.shape[0]/2):
            print('Min_val ligger i kanten')
        else:
            temp_image[:, :, :] = image[:, :, :]
            if ch == 3:
                temp_image[posy - int(filter_img.shape[1]/2):posy + int(filter_img.shape[1]/2)+1, posx - int(filter_img.shape[0]/2):posx + int(filter_img.shape[0]/2)+1, :] = filter_img[:, :, :]
                plt.subplot(5, 2, 9)
                plt.imshow(temp_image)
                plt.title('Original Image with maximum occlusion (over all channels)')
            else:
                temp_image[posy - int(filter_img.shape[1]/2):posy + int(filter_img.shape[1]/2)+1, posx - int(filter_img.shape[0]/2):posx + int(filter_img.shape[0]/2)+1, ch] = filter_img[:, :, ch]
                plt.subplot(5, 2, 2*ch+3)
                plt.imshow(temp_image)
                plt.title('Original Image with maximum occlusion (over channel '+str(ch)+')')

    plt.subplot(5, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')

    #plt.subplot(3, 3, 2)
    #plt.imshow(final_filter_images[3])
    #plt.title('Original Image with occlusion')

    plt.subplot(5, 2, 10)
    plt.imshow(predictions[3])
    plt.title('Predictions with occlusion of all channels')
    plt.colorbar()

    plt.subplot(5, 2, 4)
    plt.imshow(predictions[0])
    plt.title('Predictions with occlusion of first channel')
    plt.colorbar()

    plt.subplot(5, 2, 6)
    plt.imshow(predictions[1])
    plt.title('Predictions with occlusion of second channel')
    plt.colorbar()

    plt.subplot(5, 2, 8)
    plt.imshow(predictions[2])
    plt.title('Predictions with occlusion of third channel')
    plt.colorbar()
    plt.show()

def saliencyMap(model, image, label):
    import sys
    caffe_root = '/home/sukrit/Desktop/caffe_latest/'
    sys.path.insert(0, caffe_root + 'python')
    import caffe

    true_class = np.argmax(label)
    true_pred_full = model.predict(image.reshape(1, int(image.shape[0]), int(image.shape[1]), int(image.shape[2])), verbose=0)
    predicted_class = np.argmax(true_pred_full)
    if true_class != predicted_class:
        print('Fel prediction!')
        return
    visualize_saliency(model,layer_idx='dense_layer' , filter_indices=label, seed_input=image, backprop_modifier=None, \
                       grad_modifier="absolute")
def maximalPatches(model, x_test, img_x, img_y, layer_name, filter_nr, neuron_index_x, neuron_index_y,num_NN):
    #Current version only works for model= FinalGTSRB_model.h5
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = np.asarray([intermediate_layer_model.predict(i.reshape(1, img_x, img_y, 3)) for i in x_test])
    intermediate_output = intermediate_output[:,0,:,:,:]
    num_of_filters = intermediate_output.shape[3]
    if num_of_filters-1 < filter_nr:
        print('Fel filter nummer!')
        return
    if layer_name == 'conv2d_1':
        if neuron_index_y > 27 or neuron_index_x > 27:
            print('Fel neuron index!')
            return
        original_image_crop = [neuron_index_x, neuron_index_y, neuron_index_x + 4,
                               neuron_index_y + 4]  # Inclusive the edges
    if layer_name == 'conv2d_2':
        if neuron_index_y > 23 or neuron_index_x > 23:
            print('Fel neuron index!')
            return
        original_image_crop = [neuron_index_x, neuron_index_y, neuron_index_x + 8,
                               neuron_index_y + 8]  # Inclusive the edges    if layer_name == 'conv2d_3':  #Obs pooling
    if layer_name == 'conv2d_3':   #Obs Pooling!
        if neuron_index_y > 9 or neuron_index_x > 9:
            print('Fel neuron index!')
            return
        original_image_crop = [2*neuron_index_x, 2*neuron_index_y, 2*neuron_index_x + 13,
                               2*neuron_index_y + 13]  # Inclusive the edges
    if layer_name == 'conv2d_4':   #Obs Pooling!
        if neuron_index_y > 7 or neuron_index_x > 7:
            print('Fel neuron index!')
            return
        original_image_crop = [2*neuron_index_x, 2*neuron_index_y, 2*neuron_index_x + 17,
                               2*neuron_index_y + 17]  # Inclusive the edges

    neuron_activations = np.zeros(len(x_test))
    for img_nr in range(len(x_test)):
            actMap = intermediate_output[img_nr,:,:,filter_nr]
            neuron_activations[img_nr] = actMap[neuron_index_x, neuron_index_y]

    fig = plt.figure(figsize=(16, 16))
    for i in range(num_NN):
        index = np.argmax(neuron_activations)
        neuron_activations[index] = - np.power(10,20)
        #plt.imshow(x_test[index])
        #plt.show()
        plt.subplot(1, num_NN, i+1)
        plt.imshow(x_test[index][original_image_crop[0]:original_image_crop[2]+1, original_image_crop[1]:original_image_crop[3]+1, :])
        plt.title('Patch of image x_test[' + str(index) + '],\n that maximized neuron ('+str(neuron_index_x)+','+str(neuron_index_y)+')\n after filter ' + str(filter_nr) + 'in layer'+layer_name)
    plt.savefig('MaximalPatches.png')
    plt.show()

def tsne3D(intermediate_output, y_test):
    N = len(y_test[0])
    intermediate_output = intermediate_output[:, 0, :]
    # First do PCA to reduce complexity, to dim=50
    pca = decomposition.PCA(n_components=50)
    pca_result = pca.fit_transform(intermediate_output)

    tsne = manifold.TSNE(n_components=3, verbose=0, perplexity=40, n_iter=500)
    # tsne_results = tsne.fit_transform(intermediate_output,y_test)    #Withoout PCA: Converts the feature-space vector to a 3D-vector (len(intermediate_output), 3)
    tsne_results = tsne.fit_transform(pca_result, y_test)  # With PCA: Converts the feature-space vector to a 3D-vector (len(intermediate_output), 3)

    x = tsne_results[:, 0]
    y = tsne_results[:, 1]
    z = tsne_results[:, 2]
    fig = matplotlib.pyplot.figure()
    ax = Axes3D(fig)
    y_from_cath = np.zeros(len(y_test))
    for i in range(len(y_test)):
        y_from_cath[i] = np.argmax(y_test[i])

    #for i in range(len(tsne_results)):
    #    y_temp = np.argmax(y_test[i])
    #    print('Class nr: '+ str(y_temp)+ ', which corresponds to the 3D-vector: '+str(tsne_results[i]))

    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0, N, N + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    scat = ax.scatter(x, y, z, c=y_from_cath, cmap=cmap, norm=norm)
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Colour as a function of class')
    plt.title('T-Sne visualization of the feature space')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig('tsne.png')
    plt.show()

def weightDistribution(model):
    i=1
    fig = plt.figure(figsize=(16, 16))
    for arr in [0,2,5,7,12,14,16]:  #Array of indeces for the layers, see model.summary
        weights = model.layers[arr].get_weights()[0]
        biases = model.layers[arr].get_weights()[1]
        print(biases)
        if arr in [0,2,5,7]:  #If conv_layer
            weights_filter = np.empty((weights.shape[3], weights.shape[0]*weights.shape[1]*weights.shape[2]))   #(Num_filter, Num_weights)
            for j in range(weights.shape[3]):
                weights_filter[j] = weights[:,:,:,j].flatten()
                weights_filter[j] = weights_filter[j]/(np.max(np.abs(weights_filter[j])))  # Normalize
            layer_name = 'Convulution layer '+str(i)
            plt.subplot(2, 4, i)
            if arr == 0:
                bins = 30
            else:
                bins=100
            plt.hist(weights_filter.flatten(), bins=bins, density=True);
        else:
            weights = weights.flatten()  # Flattens all dimensions into 1D (len, )
            weights = weights / (np.max(np.abs(weights)))  # Squashes all weigths to [-1,1]
            layer_name = 'Fully connected layer ' + str(i - 4)
            plt.subplot(2, 4, i)
            plt.hist(weights, bins=100, density=True);
        i+=1
        plt.title(layer_name)
    #plt.savefig('Weight_distribution.png')
    plt.show()

def deadNeurons(model, x_test, threshhold):
    img_x = x_test[0].shape[0]
    img_y = x_test[0].shape[1]
    num_dead_neurons = list()
    num_alive_neurons = list()
    tot_num_neurons = 0
    for layer_name in ['conv2d_1', 'conv2d_2', 'max_pooling2d_1', 'conv2d_3', 'conv2d_4', 'max_pooling2d_2', 'dense_1', 'dense_2']:
        intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
        intermediate_output = np.asarray([intermediate_layer_model.predict(i.reshape(1, img_x, img_y, 3)) for i in x_test])
        if layer_name in ['dense_1', 'dense_2']:
            intermediate_output = intermediate_output[:, 0, :]  #Gives correct output shape
        else:
            intermediate_output = intermediate_output[:, 0, :, :, :]  #Gives correct output shape
        dead_neurons = np.zeros(intermediate_output[0].shape)
        tot_num_neurons += np.sum(dead_neurons == 0)
        print(tot_num_neurons)
        for img_num in range(len(x_test)):
            temp = intermediate_output[img_num] > threshhold
            dead_neurons = dead_neurons + temp  #Sets neurons that are under the threshold to zero, rest to one and adds index-wise
        #always_live_neurons = dead_neurons == len(x_test)   #Counts how many neurons that are always active
        if layer_name not in ['dense_1', 'dense_2']:
            for i in range(dead_neurons.shape[2]):
                num_dead_neurons.append(np.sum(dead_neurons[:,:,i] == 0))  # Total number of dead neurons in a layer
                #num_alive_neurons.append(np.sum(always_live_neurons[:,:,i]))  # Total number of always alive neurons in a layer
                #print(num_dead_neurons)
                #print(num_alive_neurons)
                #plt.imshow(always_live_neurons[:,:,i])
                #plt.imshow(dead_neurons[:,:,i])   #Shows how many times a neuron has been activated in a given layer
                #plt.colorbar()
                #plt.title('Number of times each neuron has been activated')
                #plt.show()
        else:
            num_dead_neurons.append(np.sum(dead_neurons == 0))   #Total number of dead neurons in a layer
            #num_alive_neurons.append(np.sum(always_live_neurons))  #Total number of always alive neurons in a layer
            #print(num_dead_neurons)
            #print(num_alive_neurons)
    frac_dead_neurons = np.sum(np.asarray(num_dead_neurons))/tot_num_neurons
    #frac_alive_neurons = np.sum(np.asarray(num_alive_neurons))/tot_num_neurons
    #print(frac_dead_neurons)
    #print(frac_alive_neurons)


    '''
    ###################### Result, threshold=0 and 10000 images
    Result: Total 68736 neurons. Zero are always active, 23 are always dead. Of those 22 are in filter number 2 in the first conv_layer, and 1 is in the
    sixth filter in the 1st conv_layer -> maybe implies that those filters have very small or negative weights, or large biases?
    Total: only 0.334613594 promille is dead!
    
    According to stanford this might imply that our learning rate is good!
    
    
    These are the biases of the first conv_layer:
    [ 0.0539727   0.06004082  0.04712844 -0.00737322  0.03799572  0.0394327
      0.05977536  0.05732208  0.08614857  0.05904264  0.05366753 -0.00089491
      0.05552064  0.02075238  0.032813    0.062197  ]
      
      Noticable is that the biases in the other layers are on the order of -3/-4 -> these layer 1 biases are a lot higher
    '''







