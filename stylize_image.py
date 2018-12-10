"""
Used to load and apply a trained faststyle model to an image in order to
stylize it.

File author: TJ Park
Date: April 2017
"""

import tensorflow as tf
import numpy as np
from fastStyleNet import transform
from PIL import Image
import scipy.misc
import os, ast

flags = tf.app.flags
flags.DEFINE_string('input_img', None, 'Directory of TFRecords training data.')
flags.DEFINE_string('output_dir', None, 'Directory of style image data set.')
flags.DEFINE_string('checkpoint', None, 'Checkpoint to load the model from')
flags.DEFINE_integer('num_styles', None, 'Number of styles the model was trained on.')
flags.DEFINE_string('output_basename', 'N-style', 'Output base name.')
flags.DEFINE_string('which_styles', None,
                    'Which styles to use. This is either a Python list or a '
                    'dictionary. If it is a list then a separate image will be '
                    'generated for each style index in the list. If it is a '
                    'dictionary which maps from style index to weight then a '
                    'single image with the linear combination of style weights '
                    'will be created. [0] is equivalent to {0: 1.0}.')
flags.DEFINE_string('upsample_method', 'resize', """Either deconvolution as in the original paper,
                        or the resize convolution method. The latter seems
                        superior and does not require TV regularization through
                        beta.""")

FLAGS = flags.FLAGS


def _load_checkpoint(sess, checkpoint):
    """
    Loads a checkpoint file into the session.
    :param sess:
    :param checkpoint:
    :return:

    """
    model_saver = tf.train.Saver(tf.global_variables())
    # checkpoint = os.path.expanduser(checkpoint)
    # if tf.gfile.IsDirectory(checkpoint):
    #     checkpoint = tf.train.latest_checkpoint(checkpoint)
    #     print('loading latest checkpoint file: {}'.format(checkpoint))
    #     model_saver.restore(sess, FLAGS.checkpoint)


    model_saver.restore(sess, FLAGS.checkpoint)

def _describe_style(which_styles):
    """
    Returns a string describing a linear combination of styles.
    """
    def _format(v):
        formatted = str(int(round(v * 1000.0)))
        while len(formatted) < 3:
            formatted = '0' + formatted
        return formatted

    values = []
    for k in sorted(which_styles.keys()):
        values.append('%s_%s' % (k, _format(which_styles[k])))
    return '_'.join(values)


def _style_mixture(which_styles, num_styles):
    """
    Returns a 1-D array mapping style indexes to weights.
    """
    if not isinstance(which_styles, dict):
        raise ValueError('Style mixture must be a dictionary.')
    mixture = np.zeros([num_styles], dtype=np.float32)
    for index in which_styles:
        mixture[index] = which_styles[index]
    return mixture


def _multiple_images(input_image, which_styles, output_dir):
    """
    Stylizes an image into a set of styles and writes them to disk.
    :param input_image:
    :param which_styles:
    :param output_dir:
    :return:
    """

    with tf.device('/cpu:0'):
        input = tf.concat([input_image for _ in range(len(which_styles))], 0)
        print(input)
        with tf.variable_scope('styleNet'):
            stylized_images = transform(input, tf.constant(which_styles), FLAGS.num_styles, norm='cond')
        # print("=============================")
        # for v in tf.global_variables():
        #     print(v)

        with tf.Session() as sess:
            _load_checkpoint(sess, FLAGS.checkpoint)

            stylized_images = sess.run(stylized_images)
            for which, stylized_image in zip(which_styles, stylized_images):
                img_out = np.squeeze(stylized_image)
                output_path = '{}/{}_{}.png'.format(output_dir, FLAGS.output_basename, which)
                scipy.misc.imsave(output_path, img_out)


def _multiple_styles(input_image, which_styles, output_dir):
    """
    Stylizes image into a linear combination of styles and writes to disk.
    """
    with tf.device('/cpu:0'):
        mixture = _style_mixture(which_styles, FLAGS.num_styles)
        with tf.variable_scope('styleNet'):
            stylized_images = transform(input_image, tf.constant(mixture), FLAGS.num_styles, norm='weight')
            print(stylized_images)

        with tf.Session() as sess:
            _load_checkpoint(sess, FLAGS.checkpoint)

            stylized_image = stylized_images.eval()
            output_path = os.path.join(output_dir, '%s_%s.png' % (FLAGS.output_basename, _describe_style(which_styles)))
            img_out = np.squeeze(stylized_image)
            scipy.misc.imsave(output_path, img_out)


if __name__ == '__main__':

    with tf.device('/cpu:0'):
        # Command-line argument parsing.
        input_img_path = FLAGS.input_img
        output_dir = FLAGS.output_dir
        upsample_method = FLAGS.upsample_method
        which_styles = FLAGS.which_styles
        num_styles = FLAGS.num_styles

        # Read + preprocess input image.
        img_arr = np.asarray(Image.open(input_img_path).convert('RGB'), dtype=np.float32)
        image = img_arr.reshape((1,) + img_arr.shape)

        output_dir = os.path.expanduser(FLAGS.output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if which_styles is None:
            which_styles = [idx for idx in range(num_styles)]
        else:
            which_styles = ast.literal_eval(FLAGS.which_styles)

        print(which_styles)

        if isinstance(which_styles, list):
            _multiple_images(image, which_styles, output_dir)
        elif isinstance(which_styles, dict):
            _multiple_styles(image, which_styles, output_dir)
        else:
            raise ValueError('--which_styles must be either a list of style indexes '
                             'or a dictionary mapping style indexes to weights.')

        print 'Done.'
        # which_styles = ast.literal_eval(FLAGS.which_styles)
        # # Create the graph.
        # with tf.variable_scope('styleNet'):
        #     X = tf.placeholder(tf.float32, shape=image.shape, name='input')
        #     Y = transform(X, tf.constant(which_styles), FLAGS.num_styles, upsample_method)
        #
        # # Saver used to restore the model to the session.
        # saver = tf.train.Saver()
        #
        # # Filter the input image.
        # with tf.Session() as sess:
        #     print 'Loading up model...'
        #     saver.restore(sess, checkpoint)
        #     print 'Evaluating...'
        #     img_out = sess.run(Y, feed_dict={X: image})
        #
        # # Postprocess + save the output image.
        # print 'Saving image.'
        # img_out = np.squeeze(img_out)
        # scipy.misc.imsave(output_dir+'/'+output_basename+'.jpg', img_out)
        #
        # print 'Done.'
