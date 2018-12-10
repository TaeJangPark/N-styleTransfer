"""
Training is done by finetuning the instance norm parameters of a pre-trained
N-styles style transfer model.

File author: TJ park
Date: April 2017
"""

import os, ast
import tensorflow as tf
import numpy as np
import vgg16
from fastStyleNet import transform
import datapipe
import utils
import losses
from PIL import Image
import scipy
import scipy.misc


CONTENT_WEIGHTS = '{"vgg/conv3_3": 1.0}'
STYLE_WEIGHTS = ('{"vgg/conv1_2": 5.0, "vgg/conv2_2": 5.0, "vgg/conv3_3": 5.0, "vgg/conv4_3": 5.0}')


flags = tf.app.flags
flags.DEFINE_string('train_dir', None, 'Directory of TFRecords training data.')
flags.DEFINE_string('style_dataset', None, 'Directory of style image data set.')
flags.DEFINE_string('model_name', "N-style", 'Name of model being trained.')
flags.DEFINE_string('style_coefficients', None,'Scales the style weights conditioned on the style image.')
flags.DEFINE_string('content_weights', CONTENT_WEIGHTS,'Content weights')
flags.DEFINE_string('style_weights', STYLE_WEIGHTS, 'Style weights')
flags.DEFINE_string('run_name', None, 'Name of log directory within the Tensoboard directory (./summaries). If not set, will use --model_name to create a unique directory.')
flags.DEFINE_string('checkpoint', None, 'Checkpoint file for the pretrained model.')
flags.DEFINE_integer('num_styles', None, 'The number of styles.')
flags.DEFINE_integer('image_size', 256, 'Image size.')
flags.DEFINE_integer('batch_size', 4, 'Batch size for training.')
flags.DEFINE_integer('n_epochs', 2, 'epoch_size.')
flags.DEFINE_integer('num_pipe_buffer', 4000, """Number of images loaded into RAM in pipeline.
                        The larger, the better the shuffling, but the more RAM
                        filled, and a slower startup.""")
flags.DEFINE_integer('train_steps', None, 'Number of training steps.')
flags.DEFINE_integer('save_summaries_secs', 15, 'Frequency at which summaries are saved, in seconds.')
flags.DEFINE_integer('save_interval_secs', 15, 'Frequency at which the model is saved, in seconds.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_float('clip_gradient_norm', 0, 'Clip gradients to this norm')
flags.DEFINE_string('upsample_method', 'resize', """Either deconvolution as in the original paper,
                        or the resize convolution method. The latter seems
                        superior and does not require TV regularization through beta.""")
FLAGS = flags.FLAGS



def main(unused_agrv = None):
    """main

    :param args:
        argparse.Namespace object from argparse.parse_args().
    """
    # Unpack command-line arguments.
    train_dir = FLAGS.train_dir
    style_dataset = FLAGS.style_dataset
    model_name = FLAGS.model_name
    preprocess_size = [FLAGS.image_size, FLAGS.image_size]
    batch_size = FLAGS.batch_size
    n_epochs = FLAGS.n_epochs
    run_name = FLAGS.run_name
    checkpoint = FLAGS.checkpoint
    learn_rate = FLAGS.learning_rate
    content_weights = FLAGS.content_weights
    style_weights = FLAGS.style_weights
    num_pipe_buffer = FLAGS.num_pipe_buffer
    style_coefficients = FLAGS.style_coefficients
    num_styles = FLAGS.num_styles
    train_steps = FLAGS.train_steps
    upsample_method = FLAGS.upsample_method

    # Setup input pipeline (delegate it to CPU to let GPU handle neural net)
    files = tf.train.match_filenames_once(train_dir + '/train-*')
    style_files = tf.train.match_filenames_once(style_dataset)
    print("style %s" % style_files)

    with tf.variable_scope('input_pipe'), tf.device('/cpu:0'):
        _, style_labels, style_grams = datapipe.style_batcher(style_files, batch_size,
                                                          preprocess_size, n_epochs,
                                                          num_pipe_buffer)
        batch_op = datapipe.batcher(files, batch_size, preprocess_size, n_epochs, num_pipe_buffer)

    """ Set up the style coefficients """
    if style_coefficients is None:
        style_coefficients = [1.0 for _ in range(num_styles)]
    else:
        style_coefficients = ast.literal_eval(style_coefficients)
    if len(style_coefficients) != num_styles:
        raise ValueError('number of style coeffients differs from number of styles')
    style_coefficient = tf.gather(tf.constant(style_coefficients), style_labels)

    """ Set up weight of style and content image """
    content_weights = ast.literal_eval(content_weights)
    style_weights = ast.literal_eval(style_weights)
    style_weights = dict([(key, style_coefficient * val) for key, val in style_weights.iteritems()])

    target_grams = []
    for name, val in style_weights.iteritems():
        target_grams.append(style_grams[name])

    # Alter the names to include a name_scope that we'll use + output suffix.
    loss_style_layers = []
    loss_style_weights = []
    loss_content_layers = []
    loss_content_weights = []
    for key, val in style_weights.iteritems():
        loss_style_layers.append(key + ':0')
        loss_style_weights.append(val)
    for key, val in content_weights.iteritems():
        loss_content_layers.append(key + ':0')
        loss_content_weights.append(val)

    # Load in image transformation network into default graph.
    shape = [batch_size] + preprocess_size + [3]
    with tf.variable_scope('styleNet'):
        X = tf.placeholder(tf.float32, shape=shape, name='input')
        Y = transform(X, style_labels, num_styles, upsample_method)
        print(Y)

    # Connect vgg directly to the image transformation network.
    with tf.variable_scope('vgg'):
        vggnet = vgg16.vgg16(Y)

    # Get the gram matrices' tensors for the style loss features.
    input_img_grams = utils.get_grams(loss_style_layers)

    # Get the tensors for content loss features.
    content_layers = utils.get_layers(loss_content_layers)

    # Create loss function
    content_targets = tuple(tf.placeholder(tf.float32,
                            shape=layer.get_shape(),
                            name='content_input_{}'.format(i))
                            for i, layer in enumerate(content_layers))

    cont_loss = losses.content_loss(content_layers, content_targets, loss_content_weights)
    style_loss = losses.style_loss(input_img_grams, target_grams, loss_style_weights)
    loss = cont_loss + style_loss
    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('style_loss', style_loss)
        tf.summary.scalar('content_loss', cont_loss)

    # We do not want to train VGG, so we must grab the subset.
    other_vars = [var for var in tf.get_variable_scope('styleNet')
                  if 'CondInstNorm' not in var.name]

    train_vars = [var for var in tf.get_variable_scope('styleNet')
                  if 'CondInstNorm' in var.name]

    # Setup step + optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss, global_step, train_vars)

    # Setup subdirectory for this run's Tensoboard logs.
    if not os.path.exists('./summaries/train/'):
        os.makedirs('./summaries/train/')
    if run_name is None:
        current_dirs = [name for name in os.listdir('./summaries/train/')
                        if os.path.isdir('./summaries/train/' + name)]
        name = model_name + '0'
        count = 0
        while name in current_dirs:
            count += 1
            name = model_name + '{}'.format(count)
        run_name = name

    # Savers and summary writers
    if not os.path.exists('./training'):  # Dir that we'll later save .ckpts to
        os.makedirs('./training')
    if not os.path.exists('./models'):  # Dir that save final models to
        os.makedirs('./models')


    saver = tf.train.Saver()
    saver_n_stylee = tf.train.Saver(other_vars)
    final_saver = tf.train.Saver(train_vars)
    merged = tf.summary.merge_all()
    full_log_path = './summaries/train/' + run_name
    train_writer = tf.summary.FileWriter(full_log_path, tf.Session().graph)

    # We must include local variables because of batch pipeline.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Begin training.
    print 'Starting training...'
    with tf.Session() as sess:
        # Initialization
        sess.run(init_op)
        vggnet.load_weights(vgg16.checkpoint_file(), sess)
        saver_n_stylee.restore(sess, checkpoint)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                current_step = sess.run(global_step)
                batch = sess.run(batch_op)

                # Collect content targets
                content_data = sess.run(content_layers, feed_dict={Y: batch})
                feed_dict = {X: batch, content_targets: content_data }

                if (current_step % 1000 == 0):
                    # Save a checkpoint
                    save_path = 'training/' + model_name + '.ckpt'
                    saver.save(sess, save_path, global_step=global_step)
                    summary, _, loss_out, c_loss, s_loss = sess.run([merged, optimizer, loss, cont_loss, style_loss],
                                                    feed_dict=feed_dict)
                    train_writer.add_summary(summary, current_step)
                    print current_step, loss_out, c_loss, s_loss

                elif (current_step % 10 == 0):
                    # Collect some diagnostic data for Tensorboard.
                    summary, _, loss_out, c_loss, s_loss = sess.run([merged, optimizer, loss, cont_loss, style_loss], feed_dict=feed_dict)
                    train_writer.add_summary(summary, current_step)

                    # Do some standard output.
                    # if (current_step % 1000 == 0):
                    print current_step, loss_out, c_loss, s_loss
                else:
                    _, loss_out = sess.run([optimizer, loss],
                                           feed_dict=feed_dict)

                # Throw error if we reach number of steps to break after.
                if current_step == train_steps:
                    print('Done training.')
                    break
        except tf.errors.OutOfRangeError:
            print('Done training.')
        finally:
            # Save the model (the image transformation network) for later usage
            # in predict.py
            final_saver.save(sess, 'models/' + model_name + '_final.ckpt', write_meta_graph=False)

            coord.request_stop()

        coord.join(threads)


if __name__ == "__main__":
    main()
