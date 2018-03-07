# Variables written in caps must be configured
# Omitting images without any traffic signes specified in gt.csv

import tensorflow as tf
import pandas as pd
import io
import numpy as np
from tensorflow.models.research.object_detection.utils import dataset_util
from PIL import Image

def create_tf_entry(label_and_data_info):

    height = 800 # Image height
    width = 1360 # Image width
    filename = label_and_data_info[0] # Filename of the image. Empty if image is not from file
    img = Image.open(IMAGE_FOLDER + filename.decode())

    b = io.BytesIO()
    img.save(b, 'PNG')

    encoded_image_data = b.getvalue() # Encoded image bytes
    image_format = b'png' # b'jpeg' or b'png'

    xmins = [x/width for x in label_and_data_info[1]] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [x/width for x in label_and_data_info[2]] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [y/height for y in label_and_data_info[3]] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [y/height for y in label_and_data_info[4]] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = label_and_data_info[5] # List of string class name of bounding box (1 per box)
    classes = label_and_data_info[6] # List of integer class id of bounding box (1 per box)

    tf_label_and_data = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_label_and_data

IMAGE_FOLDER = 'TestGTSDB/' # change here
GT_LOCATION = 'TestGTSDB/gt.txt' # here
OUTPUT_PATH = 'TestGTSDB.record' # and here.

writer = tf.python_io.TFRecordWriter(OUTPUT_PATH)


raw_data = pd.read_csv(GT_LOCATION, sep=';', header=None, names = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'ClassID'])

i = 0
prev_file = ''
all_data_and_label_info = []
for filename in raw_data['filename']:
    if filename != prev_file:
        if i != 0:
            all_data_and_label_info.append(temp_data)

        temp_data = ([str.encode(filename), [raw_data['xmin'][i] ],[raw_data['xmax'][i]],[raw_data['ymin'][i]],[raw_data['ymax'][i]],
                   [str.encode(str(raw_data['ClassID'][i]))], [raw_data['ClassID'][i] + 1]])
    else:
        temp_data[1].append(raw_data['xmin'][i])
        temp_data[2].append(raw_data['xmax'][i])
        temp_data[3].append(raw_data['ymin'][i])
        temp_data[4].append(raw_data['ymax'][i])
        temp_data[5].append(str.encode(str(raw_data['ClassID'][i])))
        temp_data[6].append(raw_data['ClassID'][i] + 1)

    prev_file = filename
    i = i + 1

all_data_and_label_info.append(temp_data)

i = 0
for data_and_label_info in all_data_and_label_info:
    tf_entry = create_tf_entry(data_and_label_info)
    writer.write(tf_entry.SerializeToString())
    i = i +1
    if i % 20 == 0:
        print(str(i) + ' images saved as TF entries.')

print('Number of images in TFRecord: ' + str(i))
writer.close()
