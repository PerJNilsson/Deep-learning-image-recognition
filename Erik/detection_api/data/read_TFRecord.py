'''import tensorflow as tf


input_path = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/data/GTSDB_training.record'
reader = tf.TFRecordReader()

filename_queue = tf.train.string_input_producer([input_path], num_epochs=1)
serialized_example = reader.read(filename_queue)
#print(serialized_example)'''
import tensorflow as tf
from google.protobuf.json_format import MessageToJson

file = "./TestGTSDB.record"
fileNum=1
for example in tf.python_io.tf_record_iterator(file):
    jsonMessage = MessageToJson(tf.train.Example.FromString(example))
    with open("./TestTFResults/image_{}".format(fileNum),"w") as text_file:
        print(jsonMessage,file=text_file)
    fileNum+=1
    if fileNum == 4:
        break
