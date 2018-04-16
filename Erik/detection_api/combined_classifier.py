''' Script that will use Faster R-CNN to find bounding box and an algoritm
trained on GTSRB for classification'''

import tensorflow as tf
import numpy as np
import os, glob
from PIL import Image


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




def GTSRBClassifier(bbox, img):
    # function that will crop image and classify
    width, height = img.size
    cropp_tuple = (bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height)
    cropped_img = img.crop(cropp_tuple)
    cropped_img.save(PATH_TO_SAVE + 'x.png')
    sign_class = 1
    return sign_class


PATH_TO_MODEL = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/fine_tuned_model/cloud/180307_2-80000/frozen_inference_graph.pb'
PATH_TO_DATA = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/data/TestGTSDB/'
PATH_TO_SAVE = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/data/results/cloud/combined_test/'
SCORE_THRESHOLD = 0.5

classifier = GTSDBClassifier()
all_imgs_paths = glob.glob(os.path.join(PATH_TO_DATA, '*.png'))

for path in all_imgs_paths[0:2]:
    img = Image.open(path)
    result = classifier.get_classification(img)

    for i in range(0, len(result[1][0])):
        if result[1][0][i] > SCORE_THRESHOLD:

            sign_class = GTSRBClassifier(result[0][0][i], img)