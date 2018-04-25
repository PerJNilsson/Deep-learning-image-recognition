''' Script that will use Faster R-CNN to find bounding box and an algoritm
trained on GTSRB for classification'''

import tensorflow as tf
import numpy as np
import os, glob
from PIL import Image
import csv
from skimage import transform
from keras.models import load_model


class GTSDBClassifier(object):
    def __init__(self):

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        return boxes, scores, classes, num




def prepare_classification(bbox, img):
    # function that will crop image and classify
    width, height = img.size
    crop_tuple = (bbox[1]*width, bbox[0]*height, bbox[3]*width, bbox[2]*height)
    cropped_img = np.array(img.crop(crop_tuple))
    cropped_img = basic_preprocess(cropped_img)
    return cropped_img, crop_tuple

def GTSRBClassifier(imgs):
    model = load_model(H5_LOCATION)
    prediction = model.predict_classes(imgs)
    return prediction

# Preprocessing with only crop and standard size
def basic_preprocess(img): # TODO - ensure same preprocessing as when training
    IMG_SIZE = 32
    # Central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2: centre[0] + min_side // 2,
          centre[1] - min_side // 2: centre[1] + min_side // 2,
          :]

    # Rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    return img

PATH_TO_MODEL = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/fine_tuned_model/cloud/180307_2-150000/frozen_inference_graph.pb'
PATH_TO_DATA = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/data/TestGTSDB/'
PATH_TO_SAVE = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/data/' \
               'results/cloud/combined_test/result_0_6.csv'
H5_LOCATION = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/models/FinalGTSRB_model.h5'
SCORE_THRESHOLD = 0.6

classifier = GTSDBClassifier()
all_imgs_paths = glob.glob(os.path.join(PATH_TO_DATA, '*.png'))
all_imgs_paths.sort()
all_res = []
imgs_to_classify = []

for path in all_imgs_paths:
    img = Image.open(path)
    result = classifier.get_classification(img)
    for i in range(0, len(result[1][0])):
        if result[1][0][i] > SCORE_THRESHOLD:
            print(result[1][0][i])
            head, filename = os.path.split(path)
            img_to_classify, crop_tuple = prepare_classification(result[0][0][i], img)
            fcnn_class = int(result[2][0][i]) - 1
            all_res.append([filename, crop_tuple[0], crop_tuple[1], crop_tuple[2], crop_tuple[3],fcnn_class ,[]])
            imgs_to_classify.append(img_to_classify)

imgs_to_classify = np.asarray(imgs_to_classify, dtype='float32')
pred_classes = GTSRBClassifier(imgs_to_classify)

for i in range(0,len(pred_classes)):
    all_res[i][6] = pred_classes[i]


res_file = open(PATH_TO_SAVE, 'w')
with res_file:
    writer = csv.writer(res_file, delimiter=';')
    writer.writerows(all_res)
