import tensorflow as tf

from utils import get_record_parser

slim = tf.contrib.slim


def get_val_dataset(record_file, config):

    parser = get_record_parser(config)
    dataset = tf.data.TFRecordDataset(record_file).\
        map(parser).\
        batch(1).\
        repeat()

    return dataset


def get_train_dataset(record_file, config):

    parser = get_record_parser(config)
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
    pass
