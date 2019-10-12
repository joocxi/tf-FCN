import os

import tensorflow as tf

from trainer import train, test_iterator
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
checkpoint_dir = os.path.join(log_dir, "checkpoints")
tensorboard_dir = os.path.join(log_dir, "tensorboard")

train_record_file = os.path.join(prepro_dir, "train.tfrecords")
val_record_file = os.path.join(prepro_dir, "val.tfrecords")
data_csv = os.path.join(prepro_dir, "data.csv")
pretrained_vgg = os.path.join(log_dir, "vgg_16.ckpt")

# directory config
flags.DEFINE_string("data_dir", data_dir, "")
flags.DEFINE_string("log_dir", log_dir, "")
flags.DEFINE_string("checkpoint_dir", checkpoint_dir, "")
flags.DEFINE_string("tensorboard_dir", tensorboard_dir, "")

# file config
flags.DEFINE_string("train_record_file", train_record_file, "")
flags.DEFINE_string("val_record_file", val_record_file, "")
flags.DEFINE_string("data_csv", data_csv, "")
flags.DEFINE_string("pretrained_vgg", pretrained_vgg, "")

# mode config
flags.DEFINE_string("mode", "train", "train/preprocess")
flags.DEFINE_integer("seed", 2019, "")

# training config
flags.DEFINE_integer("image_size", 512, "")
flags.DEFINE_integer("batch_size", 32, "")
flags.DEFINE_float("lr", 3e-4, "")
flags.DEFINE_integer("shuffle_buffer", 100, "")
flags.DEFINE_integer("train_steps", 2258, "")  # we have 2258 train samples in our split
flags.DEFINE_integer("val_steps", 565, "")  # we have 565 test samples in our split
flags.DEFINE_integer("save_summary_period", 20, "")
flags.DEFINE_integer("validation_period", 500, "")
flags.DEFINE_integer("save_model_period", 500, "")
flags.DEFINE_bool("use_augment", False, "")


if not os.path.exists(prepro_dir):
    os.makedirs(prepro_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "debug":
        config.train_steps = 3
        config.val_steps = 1
        config.batch_size = 2
        config.save_summary_period = 1
        config.validation_period = 1
        config.save_model_period = 2
        train(config)
    elif config.mode == "iter":
        test_iterator(config)
    elif config.mode == "preprocess":
        preprocess(config)


if __name__ == "__main__":
    tf.app.run()
