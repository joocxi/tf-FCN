import os

import tensorflow as tf

from trainer import train
from preprocess import preprocess

flags = tf.flags

data_dir = "data"
label_dir = "LABELLED_DICOM"
mask_dir = "MASKS_DICOM"
mesh_dir = "MESHES_VTK"
patient_dir = "PATIENT_DICOM"

class_list = ["liver", "bone", "kidney"]

flags.DEFINE_string("label_dir", label_dir, "")
flags.DEFINE_string("mask_dir", mask_dir, "")
flags.DEFINE_string("mesh_dir", mesh_dir, "")
flags.DEFINE_string("patient_dir", patient_dir, "")

flags.DEFINE_list("class_list", class_list, "")
flags.DEFINE_integer("num_classes", len(class_list) + 1, "")

prepro_dir = "prepro"
log_dir = "log"

# file
train_record_file = os.path.join(prepro_dir, "train.tfrecords")
val_record_file = os.path.join(prepro_dir, "val.tfrecords")
data_csv = os.path.join(prepro_dir, "data.csv")

# directory config
flags.DEFINE_string("data_dir", data_dir, "")
flags.DEFINE_string("log_dir", log_dir, "")

# file config
flags.DEFINE_string("train_record_file", train_record_file, "")
flags.DEFINE_string("val_record_file", val_record_file, "")
flags.DEFINE_string("data_csv", data_csv, "")

# mode config
flags.DEFINE_string("mode", "train", "train/preprocess")
flags.DEFINE_integer("seed", 2019, "")


if not os.path.exists(prepro_dir):
    os.makedirs(prepro_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "preprocess":
        preprocess(config)


if __name__ == "__main__":
    tf.app.run()
