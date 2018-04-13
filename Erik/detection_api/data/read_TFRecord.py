'''import tensorflow as tf


input_path = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/data/GTSDB_training.record'
reader = tf.TFRecordReader()

filename_queue = tf.train.string_input_producer([input_path], num_epochs=1)
serialized_example = reader.read(filename_queue)
#print(serialized_example)'''
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import io
from google.protobuf.json_format import MessageToJson

file = "./TestGTSDB.record"
RESULTS_PATH = './TestTFResults/'
fileNum=1
for string_record in tf.python_io.tf_record_iterator(file):
    jsonMessage = MessageToJson(tf.train.Example.FromString(string_record))
    with open(RESULTS_PATH + "image_{}".format(fileNum),"w") as text_file:
        print(jsonMessage,file=text_file)

    example = tf.train.Example()
    example.ParseFromString(string_record)

    height = int(example.features.feature['image/height']
                                 .int64_list
                                 .value[0])

    width = int(example.features.feature['image/width']
                                .int64_list
                                .value[0])
    class_int = int(example.features.feature['image/object/class/label']
                                .int64_list
                                .value[0])
    class_text = (example.features.feature['image/object/class/text']
                                .bytes_list
                                .value[0])
    class_text = class_text.decode()

    img_string = (example.features.feature['image/encoded']
                                    .bytes_list
                                    .value[0])
    filename = (example.features.feature['image/filename']
                                        .bytes_list
                                        .value[0])
    bbox_xmin = (example.features.feature['image/object/bbox/xmin']
                                .float_list
                                .value[0])
    bbox_xmax = (example.features.feature['image/object/bbox/xmax']
                                .float_list
                                .value[0])
    bbox_ymin = (example.features.feature['image/object/bbox/ymin']
                                .float_list
                                .value[0])
    bbox_ymax = (example.features.feature['image/object/bbox/ymax']
                                .float_list
                                .value[0])

    xy = [bbox_xmin*width, bbox_ymin*height, bbox_xmax*width, bbox_ymax*height]
    fnt = ImageFont.truetype("Library/Fonts/Arial.ttf", size=20)
    reconstructed_img = Image.open(io.BytesIO(img_string))
    draw = ImageDraw.Draw(reconstructed_img)
    draw.rectangle(xy, outline='red')
    draw.text((xy[0], xy[1]), class_text + '/' + str(class_int), fill='DeepPink', font=fnt)
    reconstructed_img.save(RESULTS_PATH + filename.decode())

    fileNum+=1
    if fileNum == 4:
        break
