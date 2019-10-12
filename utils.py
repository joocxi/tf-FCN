import tensorflow as tf
import numpy as np


def get_record_parser(config, augment=False):
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

        image = tf.concat([image] * 3, axis=-1)

        if augment:
            case = tf.random_uniform(maxval=4, dtype=tf.int32, shape=[])

            def flip_lr(img): return tf.image.flip_left_right(img)

            def flip_ud(img): return tf.image.flip_up_down(img)

            def flip_rot(img): return tf.image.rot90(img)

            image = tf.case([
                (tf.equal(case, 0), lambda: flip_lr(image)),
                (tf.equal(case, 1), lambda: flip_ud(image)),
                (tf.equal(case, 2), lambda: flip_rot(image))],
                default=lambda: image)

            mask = tf.case([
                (tf.equal(case, 0), lambda: flip_lr(mask)),
                (tf.equal(case, 1), lambda: flip_ud(mask)),
                (tf.equal(case, 2), lambda: flip_rot(mask))],
                default=lambda: mask)

        return image, mask

    return parser


def get_bilinear_filter(filter_shape, upscale_factor, name):

    # filter_shape is [width, height, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    # Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            value = (1 - abs((x - centre_location) / upscale_factor)) * \
                    (1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)

    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    bilinear_weights = tf.get_variable(name=name, initializer=init, shape=weights.shape, trainable=False)
    return bilinear_weights
