import os
import subprocess

from time import time

import tensorflow as tf
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt

from utils import get_record_parser
from model import Model

slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO)


def get_val_dataset(record_file, config):

    parser = get_record_parser(config, augment=False)
    dataset = tf.data.TFRecordDataset(record_file).\
        map(parser).\
        batch(1).\
        repeat()

    return dataset


def get_train_dataset(record_file, config):

    parser = get_record_parser(config, augment=config.use_augment)
    dataset = tf.data.TFRecordDataset(record_file).\
        map(parser).\
        shuffle(config.shuffle_buffer, seed=config.seed).\
        batch(config.batch_size).\
        repeat()

    return dataset


def create_iterator(config):
    train_dataset = get_train_dataset(config.train_record_file, config)
    val_dataset = get_val_dataset(config.val_record_file, config)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle,
                                                   train_dataset.output_types,
                                                   train_dataset.output_shapes)

    train_iterator = train_dataset.make_one_shot_iterator()
    val_iterator = val_dataset.make_one_shot_iterator()
    return train_iterator, val_iterator, iterator, handle


def test_iterator(config):
    train_iterator, val_iterator, iterator, handle = create_iterator(config)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        train_handle = sess.run(train_iterator.string_handle())
        image, segment = sess.run(iterator.get_next(), feed_dict={handle: train_handle})
        print("Image shape: {}".format(image.shape))
        print("Segment shape: {}".format(segment.shape))


def train(config):
    train_iterator, val_iterator, iterator, handle = create_iterator(config)

    fcn = Model(config, iterator)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        checkpoint_path = tf.train.latest_checkpoint(config.checkpoint_dir)
        if checkpoint_path is None:
            tf.logging.info("No checkpoint found in: {} dir.".format(config.checkpoint_dir))
            tf.logging.info("Preparing pretrained VGG model in: {} dir.".
                            format(os.path.dirname(config.pretrained_vgg)))

            if not os.path.isfile(config.pretrained_vgg):
                subprocess.run("sh scripts/vgg.sh", shell=True, check=True)
            restore = slim.assign_from_checkpoint_fn(config.pretrained_vgg,
                                                     slim.get_model_variables("vgg_16"))
            restore(sess)
            tf.logging.info("Successfully loaded VGG.")
        else:
            tf.logging.info("Loading model from checkpoint: {}.".format(checkpoint_path))
            saver.restore(sess, checkpoint_path)
            global_step = int(checkpoint_path[checkpoint_path.rfind("-") + 1:])
            tf.logging.info("Successfully loaded {} at global step {}.".format(
                os.path.basename(checkpoint_path), global_step))

        writer = tf.summary.FileWriter(config.tensorboard_dir, sess.graph)

        train_handle = sess.run(train_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())

        sess.run(tf.assign(fcn.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(fcn.lr, tf.constant(config.lr, dtype=tf.float32)))

        tf.logging.info("Training...")
        start_time = time()
        for _ in tqdm(range(config.train_steps)):
            global_step = sess.run(fcn.global_step) + 1
            loss, train_op = sess.run([fcn.loss, fcn.train_op], feed_dict={handle: train_handle})

            if global_step % config.save_summary_period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(tag="train/loss", simple_value=loss)])
                writer.add_summary(loss_sum, global_step)

            if global_step % config.validation_period == 0:
                tf.logging.info("Validating after {} steps...".format(global_step))
                sess.run(tf.local_variables_initializer())
                sess.run(tf.assign(fcn.is_train, tf.constant(False, dtype=tf.bool)))

                val_losses = []
                for _ in tqdm(range(config.val_steps)):
                    val_loss, _ = sess.run([fcn.loss, fcn.update_op], feed_dict={handle: val_handle})
                    val_losses.append(val_loss)

                val_loss = np.mean(val_losses)
                val_loss_sum = tf.Summary(value=[tf.Summary.Value(tag="val/loss", simple_value=val_loss)])
                writer.add_summary(val_loss_sum, global_step)

                sess.run(tf.assign(fcn.is_train, tf.constant(True, dtype=tf.bool)))

                m_iou = sess.run(fcn.m_iou)

                m_iou_sum = tf.Summary(value=[tf.Summary.Value(tag="val/iou_acc", simple_value=m_iou)])
                writer.add_summary(m_iou_sum, global_step)

                writer.flush()
                tf.logging.info("Finished validating.")

            if global_step % config.save_model_period == 0:
                saver.save(sess, os.path.join(config.checkpoint_dir, "model"), global_step=global_step)

        writer.close()

        tf.logging.info("Training finished in {} minutes.".format(round((time() - start_time) / 60, 2)))


def predict(config):

    val_dataset = get_val_dataset(config.val_record_file, config)

    val_iterator = val_dataset.make_one_shot_iterator()

    fcn = Model(config, val_iterator)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.checkpoint_dir))

        sess.run(tf.assign(fcn.is_train, tf.constant(False, dtype=tf.bool)))

        out_dir = os.path.join(config.out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for i in tqdm(range(config.val_steps)):
            y, prediction = sess.run([fcn.y, fcn.predict])
            prediction = np.squeeze(prediction, axis=0)
            y = np.squeeze(y)

            plt.figure()
            plt.subplot(1, 2, 1, title="True mask")
            plt.imshow(y)

            plt.subplot(1, 2, 2, title="Predicted mask")
            plt.imshow(prediction)
            plt.savefig(os.path.join(out_dir,  "out_{}.png".format(i)), transparent=True)

            plt.close()
