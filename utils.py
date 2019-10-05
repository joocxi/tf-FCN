import tensorflow as tf


def get_record_parser(config):
    def parser(example):

        features = {
            "image": tf.FixedLenFeature([], tf.string),
            "mask": tf.FixedLenFeature([], tf.string)
        }
        features = tf.parse_single_example(example, features)

        shape = tf.stack([config.image_size, config.image_size, 1])

        image = tf.decode_raw(features["image"], tf.int16)
        mask = tf.decode_raw(features["mask"], tf.uint8)

        image = tf.reshape(image, shape)
        mask = tf.reshape(mask, shape)

        image = tf.cast(image, tf.float32)
        mask = tf.cast(mask, tf.int32)

        return image, mask

    return parser
