"""
Train image transformation network in conjunction with perceptual loss. Save
the image transformation network for later application.

File author: TJ park
Date: April 2017
"""
import os, ast
import tensorflow as tf
import vgg16
from fastStyleNet import transform
import datapipe
import losses


CONTENT_WEIGHTS = '{"vgg/conv3_3": 1.0}'
STYLE_WEIGHTS = ('{"vgg/conv1_2": 40.0, "vgg/conv2_2": 40.0, "vgg/conv3_3": 40.0, "vgg/conv4_3": 40.0}')


flags = tf.app.flags
flags.DEFINE_string('train_dir', None, 'Directory of TFRecords training data.')
flags.DEFINE_string('style_dataset', None, 'Directory of style image data set.')
flags.DEFINE_string('model_name', "N-style", 'Name of model being trained.')
flags.DEFINE_string('content_weights', CONTENT_WEIGHTS,'Content weights')
flags.DEFINE_string('style_weights', STYLE_WEIGHTS, 'Style weights')
flags.DEFINE_integer('num_styles', None, 'The number of styles.')
flags.DEFINE_integer('image_size', 256, 'Image size.')
flags.DEFINE_integer('batch_size', 4, 'Batch size for training.')
flags.DEFINE_integer('n_epochs', 2, 'epoch_size.')
flags.DEFINE_integer('num_pipe_buffer', 4000, """Number of images loaded into RAM in pipeline.
                        The larger, the better the shuffling, but the more RAM
                        filled, and a slower startup.""")
flags.DEFINE_integer('train_steps', 50000, 'Number of training steps.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_string('upsample_method', 'resize', """Either deconvolution as in the original paper,
                        or the resize convolution method. The latter seems
                        superior and does not require TV regularization through
                        beta.""")
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
    learn_rate = FLAGS.learning_rate
    content_weights = FLAGS.content_weights
    style_weights = FLAGS.style_weights
    num_pipe_buffer = FLAGS.num_pipe_buffer
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

    """ Set up weight of style and content image """
    content_weights = ast.literal_eval(content_weights)
    style_weights = ast.literal_eval(style_weights)

    target_grams = []
    for name, val in style_weights.iteritems():
        target_grams.append(style_grams[name])

    # Alter the names to include a namescope that we'll use + output suffix.
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
    input_img_grams = losses.get_grams(loss_style_layers)

    # Get the tensors for content loss features.
    content_layers = losses.get_layers(loss_content_layers)

    # Create loss function
    content_targets = tuple(tf.placeholder(tf.float32,
                            shape=layer.get_shape(),
                            name='content_input_{}'.format(i))
                            for i, layer in enumerate(content_layers))
    cont_loss = losses.content_loss(content_layers, content_targets, loss_content_weights)
    style_loss = losses.style_loss(input_img_grams, target_grams, loss_style_weights)
    tv_loss = losses.tv_loss(Y)
    loss = cont_loss + style_loss + tv_loss

    # We do not want to train VGG, so we must grab the subset.
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='styleNet')

    # Setup step + optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss, global_step, train_vars)

    if not os.path.exists('./models'):  # Dir that save final models to
        os.makedirs('./models')
    final_saver = tf.train.Saver(train_vars)

    # We must include local variables because of batch pipeline.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Begin training.
    print 'Starting training...'
    with tf.Session() as sess:
        # Initialization
        sess.run(init_op)
        vggnet.load_weights(vgg16.checkpoint_file(), sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                current_step = sess.run(global_step)
                batch = sess.run(batch_op)

                # Collect content targets
                content_data = sess.run(content_layers, feed_dict={Y: batch})
                feed_dict = {X: batch, content_targets: content_data }

                _, loss_out = sess.run([optimizer, loss], feed_dict=feed_dict)
                if (current_step % 10 == 0):
                    print current_step, loss_out

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
