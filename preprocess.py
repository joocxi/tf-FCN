import os

import pandas as pd
import pydicom as dicom
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from sklearn.model_selection import train_test_split


def preprocess(config):
    data = []
    parts = os.listdir(config.data_dir)
    for part in tqdm(parts):
        patient_dir = os.path.join(config.data_dir, part, config.patient_dir)
        mask_dir = os.path.join(config.data_dir, part, config.mask_dir)

        available_masks = os.listdir(mask_dir)

        for img_name in os.listdir(patient_dir):
            img_path = os.path.join(patient_dir, img_name)
            sample = {
                "Path": img_path,
            }

            for available_mask in available_masks:
                for mask_type in config.class_list:
                    if mask_type in available_mask:
                        mask_path = os.path.join(mask_dir, available_mask, img_name)
                        if sample.get(mask_type) is None:
                            sample[mask_type] = []
                        sample[mask_type].append(mask_path)

            data.append(sample)

    data = sorted(data, key=lambda x: x["Path"])
    data_df = pd.DataFrame(data)
    data_df.to_csv(config.data_csv, index=False)

    write_records(data_df, config)


def build_features(data_df, record_file, config):

    print("Creating {}..".format(record_file))

    writer = tf.python_io.TFRecordWriter(record_file)

    for idx, row in tqdm(data_df.iterrows()):

        image_ds = dicom.read_file(row["Path"])

        assert image_ds.pixel_array.dtype == np.int16
        image = image_ds.pixel_array.tostring()

        masks = []
        for i in range(len(config.class_list)):

            sub_mask_paths = row[config.class_list[i]]
            if pd.isnull(sub_mask_paths) is True:
                continue
            sub_masks = []

            for sub_mask_path in sub_mask_paths:
                sub_mask = dicom.read_file(sub_mask_path).pixel_array
                sub_masks.append(sub_mask)

            aggregated_mask = sum(sub_masks)
            aggregated_mask[aggregated_mask > 1] = 1

            assert sum(np.unique(aggregated_mask)) <= 1

            aggregated_mask[aggregated_mask == 1] = i + 1
            masks.append(aggregated_mask)

        final_mask = sum(masks)
        assert set(np.unique(final_mask)).issubset(set(c for c in range(config.num_classes)))
        assert final_mask.dtype == np.uint8

        mask = final_mask.tostring()

        feature_dict = {
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            "mask": tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask]))
        }

        features = tf.train.Features(feature=feature_dict)

        example = tf.train.Example(features=features)

        writer.write(example.SerializeToString())


def write_records(data_df, config):
    train_df, test_df = train_test_split(data_df,
                                         test_size=0.2,
                                         random_state=config.seed)

    print("Number of train samples: {}".format(len(train_df)))
    print("Number of test samples: {}".format(len(test_df)))

    build_features(train_df, config.train_record_file, config)
    build_features(test_df, config.val_record_file, config)
