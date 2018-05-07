from PIL import Image

img_list = ['/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/'
            'data/results/cloud/GTSDB-250K-eval-imgs/good-imgs/image-15.png',
            '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/'
            'data/results/cloud/GTSDB-250K-eval-imgs/good-imgs/image-29.png',
            '/Users/erikpersson/PycharmProjects/Deep-learning-image-recognition/Erik/detection_api/'
            'data/results/cloud/GTSDB-250K-eval-imgs/good-imgs/image-52.png']


def create_collage(width, height, listofimages):
    cols = 1
    rows = 3
    thumbnail_width = width
    thumbnail_height = height
    size = thumbnail_width, thumbnail_height
    new_im = Image.new('RGB', (width*cols, height*rows))
    ims = []
    for p in listofimages:
        im = Image.open(p)
        im.thumbnail(size)
        ims.append(im)
    i = 0
    x = 0
    y = 0
    for col in range(cols):
        for row in range(rows):
            print(i, x, y)
            new_im.paste(ims[i], (x, y))
            i += 1
            y += thumbnail_height
        x += thumbnail_width
        y = 0

    new_im.save("Collage3.png")

create_collage(1360, 800, img_list)