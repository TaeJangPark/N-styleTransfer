"""
This file is used for construction of the data input pipeline. It takes care of
batching and preprocessing, and can be used to repeatedly draw a fresh batch
for use in training. It utilizes TFRecords format, so data must be converted to
this beforehand. tfrecords_writer.py handles this.

File author: TJ Park
Date: April 2017
"""

import tensorflow as tf


def preprocessing(image, resize_shape):
    """Simply resizes the image.

    :param image:
        image tensor
    :param resize_shape:
        list of dimensions
    """
    if resize_shape is None:
        return image
    else:
        image = tf.image.resize_images(image, size=resize_shape, method=2)
        return image


def read_my_file_format(filename_queue, resize_shape=None):
    """Sets up part of the pipeline that takes elements from the filename queue
    and turns it into a tf.Tensor of a batch of images.

    :param filename_queue:
        tf.train.string_input_producer object
    :param resize_shape:
        2 element list defining the shape to resize images to.
    """
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example, features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64)})
    example = tf.image.decode_jpeg(features['image/encoded'], 3)
    processed_example = preprocessing(example, resize_shape)
    return processed_example


def batcher(filenames, batch_size, resize_shape=None, num_epochs=None,
            min_after_dequeue=4000):
    """Creates the batching part of the pipeline.

    :param filenames:
        list of filenames
    :param batch_size:
        size of batches that get output upon each access.
    :param resize_shape:
        for preprocessing. What to resize images to.
    :param num_epochs:
        number of epochs that define end of training set.
    :param min_after_dequeue:
        min_after_dequeue defines how big a buffer we will randomly sample
        from -- bigger means better shuffling but slower start up and more
        memory used.
        capacity must be larger than min_after_dequeue and the amount larger
        determines the maximum we will prefetch.  Recommendation:
        min_after_dequeue + (num_threads + a small safety margin) * batch_size
    """
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example = read_my_file_format(filename_queue, resize_shape)
    example = tf.to_float(example) / 255.0
    capacity = min_after_dequeue + 3 * batch_size
    example_batch = tf.train.shuffle_batch(
        [example], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch

def read_style_file_format(filename_queue, resize_shape=None):

    shuffle_queue = tf.RandomShuffleQueue(capacity=64,
                                          min_after_dequeue=32,
                                          dtypes=[tf.string],
                                          name='random_shuffle_queue')


    reader = tf.TFRecordReader()
    print("file name : {}".format(filename_queue))
    key, serialized_example = reader.read(filename_queue)
    shuffle_ops = [shuffle_queue.enqueue([serialized_example])]
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(shuffle_queue, shuffle_ops))
    shuffled_examples = shuffle_queue.dequeue()

    features = tf.parse_single_example(shuffled_examples,
                                        features={'label': tf.FixedLenFeature([], tf.int64),
                                                  'image_raw': tf.FixedLenFeature([], tf.string),
                                                  'vgg/conv1_2': tf.FixedLenFeature([64, 64], tf.float32),
                                                  'vgg/conv2_2': tf.FixedLenFeature([128, 128], tf.float32),
                                                  'vgg/conv3_3': tf.FixedLenFeature([256, 256], tf.float32),
                                                  'vgg/conv4_3': tf.FixedLenFeature([512, 512], tf.float32)})
    return features



def style_batcher(filenames, batch_size, resize_shape=None, num_epoch=None, min_after_dequeue=4000, square_crop=False):
    """
    Loads a style image at random.

    :param filenames:
            list of filenames
    :param batch_size:
            size of batches that get output upon each access.
    :param resize_shape:
            for preprocessing. What to resize images to.
    :param num_epochs:
            number of epochs that define end of training set.
    :param min_after_dequeue:
            min_after_dequeue defines how big a buffer we will randomly sample
            from -- bigger means better shuffling but slower start up and more
            memory used.
            capacity must be larger than min_after_dequeue and the amount larger
            determines the maximum we will prefetch.  Recommendation:
            min_after_dequeue + (num_threads + a small safety margin) * batch_size
    """

    vgg_layers = ['vgg/conv1_2', 'vgg/conv2_2', 'vgg/conv3_3', 'vgg/conv4_3']

    if resize_shape is None:
        raise ValueError('resize size is not define.')
    if batch_size is None:
        raise ValueError('batch size is not define.')

    with tf.name_scope('style_batcher_processing'):
        # filenames_queue = tf.train.string_input_producer(filenames,
        #                                                  num_epochs=num_epoch,
        #                                                  shuffle=True)
        filenames_queue = tf.train.string_input_producer(filenames,
                                                         shuffle=True)
        # image, label, gram = read_style_file_format(filenames_queue, resize_shape, vgg_layers)
        features = read_style_file_format(filenames_queue, resize_shape)

        image = tf.image.decode_jpeg(features['image_raw'], 3)
        resized_image = preprocessing(image, resize_shape)
        # norm_image = tf.to_float(resized_image)
        norm_image = tf.to_float(resized_image) / 255.0

        label = features['label']
        gram_matrices = [features[vgg_layer] for vgg_layer in vgg_layers]

        # capacity = min_after_dequeue + 3 * batch_size
        example_batch = tf.train.batch([norm_image, label] + gram_matrices,
                                       batch_size=batch_size)

        image_batch = example_batch[0]
        label_batch = example_batch[1]

        gram_batch = example_batch[2:]
        gram_matrices = dict([(vgg_layer, gram_matrix) for vgg_layer, gram_matrix in zip(vgg_layers, gram_batch)])
    return image_batch, label_batch, gram_matrices
