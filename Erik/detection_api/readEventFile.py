import tensorflow as tf
from PIL import Image
import io

PATH_TO_EVENT_FILE = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/' \
                     'Erik/detection_api/models/eval180307_2-150000-all/events.out.tfevents.1523964805.' \
                     'Eriks-MacBook-Pro.local'
PATH_TO_SAVE_DIR = '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/' \
                   'detection_api/data/results/cloud/GTSDB-150K-eval-imgs/'
imgs_of_interest = set()
for i in range(0,300):
    tmp_str = 'image-' + str(i)
    imgs_of_interest.add(tmp_str)


for e in tf.train.summary_iterator(PATH_TO_EVENT_FILE):
    for v in e.summary.value:
        if v.tag in imgs_of_interest:
            img_str = v.image.encoded_image_string
            stream = io.BytesIO(img_str)
            rec_img = Image.open(stream)
            rec_img.save(PATH_TO_SAVE_DIR + v.tag + '.png')