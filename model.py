import tensorflow as tf

from tensorflow.contrib.slim.nets import vgg

slim = tf.contrib.slim


class Model(object):

    def __init__(self, config, iterator=None, trainable=True):

        self.image_size = config.image_size
        self.ignored_label = config.ignored_label
        self.num_classes = config.num_classes

        self.m_iou = None
        self.loss = None
        self.pred = None

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
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')
