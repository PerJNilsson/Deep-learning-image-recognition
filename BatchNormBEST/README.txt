Num_epochs = 40
Batch_size = 30
lr = 0.001

Batch normalisation after first convulotional layer, and after two first fully connected layers
No drop out or l2 regularisation
Relu activation funtion 
Learning decay used

Preproccesing:
Crop out traffic sign
Resize to 32x32
Normalize image contrast with PIL.ImageOps.autocontrast
Equalize the image histogram with PIL.ImageOps.equalize

20/80 validation/training split 


Validation accuracy maximized after 24 epochs
RESULT
training accuracy = 100%
validation accuracy = 99.94%
test accuracy = 99.02% 