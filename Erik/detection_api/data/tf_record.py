# Variables written in caps must be configured
# Omitting images without any traffic signes specified in gt.csv

import tensorflow as tf
import pandas as pd
import io, glob, os
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
OUTPUT_PATH = 'TestGTSDB_all.record' # and here.

classes_int = list(range(1, 44))
classes_text = ['speed limit 20', 'speed limit 30', 'speed limit 50', 'speed limit 60', 'speed limit 70',
                    'speed limit 80', 'restriction ends 80', 'speed limit 100', 'speed limit 120', 'no overtaking',
                    'no overtaking (trucks)', 'priority at next intersection', 'priority road', 'give way', 'stop',
                    'no traffic both way', 'no trucks', 'no entry', 'danger', 'bend left', 'bend right', 'bend',
                    'uneven road', 'slippery road', 'road narrows', 'construction', 'traffic signal',
                    'pedestrian crossing', 'school crossing', 'cycles crossing', 'snow', 'animals', 'restriction ends',
                    'go right', 'go left', 'go straight', 'go right or straight', 'go left or straight', 'keep right',
                    'keep left', 'roundabout', 'restriction ends (overtaking)', 'restriction ends (overtaking (trucks))']
label_map = dict(zip(classes_int, classes_text))

writer = tf.python_io.TFRecordWriter(OUTPUT_PATH)


raw_data = pd.read_csv(GT_LOCATION, sep=';', header=None, names = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'ClassID'])
all_files = glob.glob(os.path.join(IMAGE_FOLDER, '*.png'))
all_files.sort()
tmp_file_list = []
for path in all_files:
   head, tail = os.path.split(path)
   tmp_file_list.append(tail)

all_files = tmp_file_list

all_data_and_label_info = []
raw_data_file = raw_data['filename'].tolist()
for name in all_files:
    if name not in raw_data_file:
        tmp_data = ([str.encode(name), [], [], [], [], [], [] ])
        all_data_and_label_info.append(tmp_data)

i = 0
prev_file = ''
for filename in raw_data['filename']:
    if filename != prev_file:
        if i != 0:
            all_data_and_label_info.append(tmp_data)

        tmp_data = ([str.encode(filename), [raw_data['xmin'][i] ],[raw_data['xmax'][i]],[raw_data['ymin'][i]],[raw_data['ymax'][i]],
                   [str.encode(label_map[(raw_data['ClassID'][i] + 1)])], [raw_data['ClassID'][i] + 1]])
    else:
        tmp_data[1].append(raw_data['xmin'][i])
        tmp_data[2].append(raw_data['xmax'][i])
        tmp_data[3].append(raw_data['ymin'][i])
        tmp_data[4].append(raw_data['ymax'][i])
        tmp_data[5].append(str.encode(label_map[(raw_data['ClassID'][i] + 1)]))
        tmp_data[6].append(raw_data['ClassID'][i] + 1)

    prev_file = filename
    i = i + 1

all_data_and_label_info.append(tmp_data)
np.random.shuffle(all_data_and_label_info)

i = 0
for data_and_label_info in all_data_and_label_info:
    tf_entry = create_tf_entry(data_and_label_info)
    writer.write(tf_entry.SerializeToString())
    i = i +1
    if i % 20 == 0:
        print(str(i) + ' images saved as TF entries.')

print('Number of images in TFRecord: ' + str(i))
writer.close()
