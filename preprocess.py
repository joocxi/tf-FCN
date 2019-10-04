
import os

import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split


def preprocess(config):
    data = []
    parts = os.listdir(config.data_dir)
    for part in parts:
        label_dir = os.path.join(config.data_dir, part, config.label_dir)
        mask_dir = os.path.join(config.data_dir, part, config.mask_dir)

        for img_name in os.listdir(label_dir):
            label_path = os.path.join(label_dir, img_name)
            sample = {
                "Path": label_path,
            }

            for mask_type in config.class_list:
                mask_path = os.path.join(mask_dir, mask_type, img_name)
                sample[mask_type] = mask_path

            data.append(sample)

    data = sorted(data, key=lambda x: x["Path"])
    data_df = pd.DataFrame(data)
    data_df.to_csv(config.data_csv, index=False)


def build_features(data_df, record_file, config):

    writer = tf.python_io.TFRecordWriter(record_file)

    # TODO: write records


def write_records(data_df, config):
    train_df, test_df = train_test_split(data_df,
                                         test_size=0.2,
                                         random_state=config.seed)

    build_features(train_df, config.train_record_file, config)
    build_features(test_df, config.val_record_file, config)
