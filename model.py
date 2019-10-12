import tensorflow as tf

from tensorflow.contrib.slim.nets import vgg

from utils import get_bilinear_filter

slim = tf.contrib.slim


class Model(object):

    def __init__(self, config, iterator=None, trainable=True):

        self.image_size = config.image_size
        self.num_classes = config.num_classes

        self.m_iou = None
        self.loss = None
        self.predict = None
        self.update_op = None

        self.global_step = tf.get_variable("global_step",
                                           shape=[],
                                           dtype=tf.int32,
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)

        self.is_train = tf.get_variable("is_train",
                                        shape=[],
                                        dtype=tf.bool,
                                        trainable=False)

        if iterator is not None:
            self.x, self.y = iterator.get_next()

            self.ready()

            if trainable:
                self.lr = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def ready(self):
        with tf.variable_scope('vgg_16'):
            with slim.arg_scope(vgg.vgg_arg_scope()):
                net = slim.repeat(self.x, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                pool3 = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                pool4 = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                pool5 = slim.max_pool2d(net, [2, 2], scope='pool5')

        with tf.variable_scope('fc'):
            net = tf.layers.conv2d(pool5, 512, 7, padding='SAME', activation=tf.nn.relu)
            net = tf.layers.dropout(net, rate=0.5, training=self.is_train)

        with tf.variable_scope('up_sampling'):
            batch_size = tf.shape(net)[0]
            output_shape = (batch_size, self.image_size, self.image_size, self.num_classes)

            # (batch_size, 16, 16, 4)
            net_score = tf.layers.conv2d(net, self.num_classes, 3, padding='SAME', activation=None)

            net_up2_kernel = get_bilinear_filter(filter_shape=[4, 4, self.num_classes, self.num_classes],
                                                 upscale_factor=2,
                                                 name="score_fr_up2")

            # (batch_size, 32, 32, 4)
            net_up2 = tf.nn.conv2d_transpose(value=net_score,
                                             filter=net_up2_kernel,
                                             output_shape=[batch_size, 32, 32, self.num_classes],
                                             strides=[1, 2, 2, 1])

            # (batch_size, 32, 32, 4)
            pool4_score = tf.layers.conv2d(pool4, self.num_classes, 1, padding='SAME', activation=None)
            pool4_fused = pool4_score + net_up2

            pool4_up2_kernel = get_bilinear_filter(filter_shape=[4, 4, self.num_classes, self.num_classes],
                                                   upscale_factor=2,
                                                   name="pool4_up2")

            # (batch_size, 64, 64, 4)
            pool4_up2 = tf.nn.conv2d_transpose(value=pool4_fused,
                                               filter=pool4_up2_kernel,
                                               output_shape=[batch_size, 64, 64, self.num_classes],
                                               strides=[1, 2, 2, 1])
            # (batch_size, 64, 64, 4)
            pool3_score = tf.layers.conv2d(pool3, self.num_classes, 1, padding='SAME', activation=None)
            pool3_fused = pool3_score + pool4_up2

            pool3_up8_kernel = get_bilinear_filter(filter_shape=[16, 16, self.num_classes, self.num_classes],
                                                   upscale_factor=8,
                                                   name="pool3_up8")

            # (batch_size, 512, 512, 4)
            up8 = tf.nn.conv2d_transpose(value=pool3_fused,
                                         filter=pool3_up8_kernel,
                                         output_shape=output_shape,
                                         strides=[1, 8, 8, 1])

        with tf.name_scope('loss'):

            self.loss = tf.losses.sparse_softmax_cross_entropy(self.y, up8)

            # (batch_size, 512, 512)
            self.predict = tf.argmax(up8, axis=-1)

            y_squeezed = tf.squeeze(self.y, axis=3)
            self.m_iou, self.update_op = tf.contrib.metrics.streaming_mean_iou(self.predict,
                                                                               y_squeezed,
                                                                               self.num_classes)
