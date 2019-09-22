from collections import namedtuple
from itertools import chain
from pathlib import Path
from PIL import Image

import io

import tensorflow as tf


ImageInfo = namedtuple(
    "ImageInfo", ["path", "class_name", "class_id", "boundingbox_path"]
)


def find_boundingbox_path(image_path):
    filename = image_path.name[:-3] + "txt"
    boundingbox_path = image_path.parent / "Label" / filename
    return boundingbox_path


def to_image_info(image_path, label_name, label_id):
    boundingbox_path = find_boundingbox_path(image_path)
    return ImageInfo(image_path, label_name, label_id, boundingbox_path)


def main():
    labels = dict(Person=1, Car=2)
    train = Path('./object_detection/smart_spy/dataset/train')
    test = Path('./object_detection/smart_spy/dataset/test')

    infos = list()
    for ds in [train, test]:
        image_path_list = chain(ds.glob("Person/*.jpg"), ds.glob("Car/*.jpg"))
        for i in image_path_list:
            info = to_image_info(i, i.parent.name, labels[i.parent.name])
            infos.append(info)

    for i in infos:
        with open(i.boundingbox_path, "r") as f:
            with tf.gfile.GFile(str(i.path), "rb") as fid:
                encoded_image_data = fid.read()
                encoded_image_io = io.BytesIO(encoded_image_data)

                image = Image.open(encoded_image_io)
                width, height = image.size
                lines = f.readlines()

                for line in lines:
                    split = line.split(" ")
                    xmin, ymin, xmax, ymax = split[-4:]
                    xmin = float(xmin)
                    ymin = float(ymin)
                    xmax = float(xmax)
                    ymax = float(ymax)

                    if not (xmin < xmax <= width):
                        print('X: ', image.size, i)
                    if not(ymin < ymax <= height):
                        print('Y: ', i)


if __name__ == "__main__":
    main()
