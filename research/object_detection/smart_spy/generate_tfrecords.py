from collections import namedtuple
from itertools import chain
from pathlib import Path
from PIL import Image

import io

import tensorflow as tf

from object_detection.utils import dataset_util

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util


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


def create_tf_example(example: ImageInfo):
    with tf.gfile.GFile(str(example.path), "rb") as fid:
        encoded_image_data = fid.read()

    encoded_image_io = io.BytesIO(encoded_image_data)
    image = Image.open(encoded_image_io)
    height, width = image.size
    filename = example.path.name.encode("utf8")
    image_format = b"jpeg"

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    with open(example.boundingbox_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            split = line.split(" ")
            xmin, ymin, xmax, ymax = split[-4:]
            class_text = " ".join(split[:-4])

            if class_text != example.class_name:
                print("Classes Divergentes")
                print(example)
                continue

            xmins.append(float(xmin) / width)
            xmaxs.append(float(xmax) / width)
            ymins.append(float(ymin) / height)
            ymaxs.append(float(ymax) / height)

            classes_text.append(class_text.encode("utf8"))
            classes.append(example.class_id)

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(filename),
                "image/source_id": dataset_util.bytes_feature(filename),
                "image/encoded": dataset_util.bytes_feature(encoded_image_data),
                "image/format": dataset_util.bytes_feature(image_format),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(
                    classes_text
                ),
                "image/object/class/label": dataset_util.int64_list_feature(classes),
            }
        )
    )
    return tf_example


def main(_):
    labels = dict(Person=1, Car=2)

    train = Path('./object_detection/smart_spy/dataset/train')
    test = Path('./object_detection/smart_spy/dataset/test')
    # train = Path("/tmp/ds/train")
    # test = Path("/tmp/ds/test")

    for ds in [train, test]:
        num_shards = 50
        output_filebase = f"./object_detection/smart_spy/dataset/tfrecords/{ds.name}/tf.record"

        image_path_list = chain(ds.glob("Person/*.jpg"), ds.glob("Car/*.jpg"))
        examples = [
            to_image_info(i, i.parent.name, labels[i.parent.name])
            for i in image_path_list
        ]

        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                tf_record_close_stack, output_filebase, num_shards
            )
            for index, example in enumerate(examples):
                tf_example = create_tf_example(example)
                output_shard_index = index % num_shards
                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

        # writer = tf.io.TFRecordWriter("./object_detection/smart_spy/tfrecords/train_dataset.full.tfrecord")
        # for example in examples:
        #     tf_example = create_tf_example(example)
        #     writer.write(tf_example.SerializeToString())
        # writer.close()

if __name__ == "__main__":
    tf.app.run()
