# Script to evaluate the output of a trained network on GTSDB

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, glob



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


def paint_box(results):
    image = Image.open(PATH_TO_DATA + results[1])
    width, height = image.size
    fnt = ImageFont.truetype('/Library/Fonts/Arial Bold.ttf', size=12)
    for i in range(len(results[0])):
        xy = [results[0][i][0][1]*width, results[0][i][0][0]*height, results[0][i][0][3]*width, results[0][i][0][2]*height]
        draw = ImageDraw.Draw(image)
        draw.rectangle(xy, outline='red')
        draw.text((xy[0], xy[1]), str(int(results[0][i][2]))+ ' ' + ('%.2f' % results[0][i][1]), fill='orange', font=fnt )
    image.save(PATH_TO_SAVE + results[1])

PATH_TO_MODEL = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/fine_tuned_model/cloud/ext_set-200000/frozen_inference_graph.pb'
PATH_TO_DATA = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/data/TestGTSDB/'
PATH_TO_SAVE = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/data/results/cloud_ext/'
SCORE_THRESHOLD = 0.5
obj1 = GTSDBClassifier()
all_imgs_paths = glob.glob(os.path.join(PATH_TO_DATA, '*.png'))
np.random.shuffle(all_imgs_paths)

all_res = []
all_imgs = []

for path in all_imgs_paths[0:2]:
    img = Image.open(path)
    res = obj1.get_classification(img)
    tmp = []
    for i in range(0, len(res[1][0])):
        print(res[1][0][i])
        if res[1][0][i] > SCORE_THRESHOLD:
            tmp.append([res[0][0][i], res[1][0][i], res[2][0][i]])

        else:
            break
    if len(tmp) > 0:
        head, tail = os.path.split(path)
        all_res.append([tmp, tail])


'''for res in all_res:
    paint_box(res)'''
