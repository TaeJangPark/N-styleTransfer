"""
Functions used for the creation of the image transformation network.

File author: TJ Park
Date: April 2017
"""

import tensorflow as tf


# TODO: For resize-convolution, what if we use strides of 1 for the
# convolution instead of upsampling past the desired dimensions? Test this.


def transform(X, style_labels, num_styles, upsample_method='deconv', norm='cond'):
    """Creates the transformation network, given dimensions acquired from an
    input image. Does this according to J.C. Johnson's specifications
    after utilizing instance normalization (i.e. halving dimensions given
    in the paper).

    :param X
        tf.Tensor with NxHxWxC dimensions.
    :param upsample_method
        values: 'deconv', 'resize'
        Whether to upsample via deconvolution, or the proposed fix of resizing
        + convolution. Description of 2nd method is available at:
            http://distill.pub/2016/deconv_checkerboard/
    """
    assert(upsample_method in ['deconv', 'resize'])
    assert (norm in ['cond', 'weight'])

    norm_func = None
    if norm is 'cond':
        norm_func = cond_inst_norm
    elif norm is 'weight':
        norm_func = weighted_inst_norm

    # Padding
    h = reflect_pad(X, 40)

    # Initial convolutional layers
    with tf.variable_scope('initconv_0'):
        h = relu(norm_func(conv2d(h, 3, 16, 9, [1, 1, 1, 1]), style_labels, num_styles))
    with tf.variable_scope('initconv_1'):
        h = relu(norm_func(conv2d(h, 16, 32, 3, [1, 2, 2, 1]), style_labels, num_styles))
    with tf.variable_scope('initconv_2'):
        h = relu(norm_func(conv2d(h, 32, 64, 3, [1, 2, 2, 1]), style_labels, num_styles))

    # Residual layers
    with tf.variable_scope('resblock_0'):
        h = res_layer(h, 64, 3, [1, 1, 1, 1], style_labels, num_styles, norm_func)
    with tf.variable_scope('resblock_1'):
        h = res_layer(h, 64, 3, [1, 1, 1, 1], style_labels, num_styles, norm_func)
    with tf.variable_scope('resblock_2'):
        h = res_layer(h, 64, 3, [1, 1, 1, 1], style_labels, num_styles, norm_func)
    with tf.variable_scope('resblock_3'):
        h = res_layer(h, 64, 3, [1, 1, 1, 1], style_labels, num_styles, norm_func)
    with tf.variable_scope('resblock_4'):
        h = res_layer(h, 64, 3, [1, 1, 1, 1], style_labels, num_styles, norm_func)

    with tf.variable_scope('upsample_0'):
        h = relu(norm_func(upconv2d(h, 64, 32, 3, [1, 2, 2, 1]), style_labels, num_styles))
    with tf.variable_scope('upsample_1'):
        h = relu(norm_func(upconv2d(h, 32, 16, 3, [1, 2, 2, 1]), style_labels, num_styles))
    with tf.variable_scope('upsample_2'):  # Not actually an upsample
        # h = scaled_tanh(norm_func(conv2d(h, 32, 3, 9, [1, 1, 1, 1]), style_labels, num_styles))
        h = tf.nn.sigmoid(norm_func(conv2d(h, 16, 3, 9, [1, 1, 1, 1]), style_labels, num_styles))

    # Create a redundant layer with name 'output'
    h = tf.identity(h, name='output')

    return h


def reflect_pad(X, padsize):
    """Pre-net padding.

    :param X
        Input image tensor
    :param padsize
        Amount by which to pad the image tensor
    """
    h = tf.pad(X, paddings=[[0, 0], [padsize, padsize], [padsize, padsize],
                            [0, 0]], mode='REFLECT')
    return h


def conv2d(X, n_ch_in, n_ch_out, kernel_size, strides, name=None, padding='SAME'):
    """Creates the convolutional layer.

    :param X
        Input tensor
    :param n_ch_in
        Number of input channels
    :param n_ch_out
        Number of output channels
    :param kernel_size
        Dimension of the square-shaped convolutional kernel
    :param strides
        Length 4 vector of stride information
    :param name
        Optional name for the weight matrix
    """
    if name is None:
        name = 'W'
    # if kernel_size % 2 == 0:
    #     raise ValueError('kernel_size is expected to be odd.')
    # pad_size = kernel_size // 2
    # padded_input = tf.pad(X, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='REFLECT')

    shape = [kernel_size, kernel_size, n_ch_in, n_ch_out]
    W = tf.get_variable(name=name,
                        shape=shape,
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
    h = tf.nn.conv2d(X,
                     filter=W,
                     strides=strides,
                     padding=padding)
    return h


def upconv2d(X, n_ch_in, n_ch_out, kernel_size, strides):
    """Resizes then applies a convolution.

    :param X
        Input tensor
    :param n_ch_in
        Number of input channels
    :param n_ch_out
        Number of output channels
    :param kernel_size
        Size of square shaped convolutional kernel
    :param strides
        Stride information
    """

    if kernel_size % 2 == 0:
        raise ValueError('kernel_size is expected to be odd.')

    # We first upsample two strides-worths. The convolution will then bring it
    # down one stride.
    new_h = X.get_shape().as_list()[1]*strides[1]**2
    new_w = X.get_shape().as_list()[2]*strides[2]**2
    upsized = tf.image.resize_images(X, [new_h, new_w], method=1)

    # pad_size = kernel_size // 2
    # padded_input = tf.pad(upsized, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='REFLECT')

    # Now convolve to get the channels to what we want.
    shape = [kernel_size, kernel_size, n_ch_in, n_ch_out]
    W = tf.get_variable(name='W',
                        shape=shape,
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
    h = tf.nn.conv2d(upsized,
                     filter=W,
                     strides=strides,
                     padding="SAME")

    return h


def deconv2d(X, n_ch_in, n_ch_out, kernel_size, strides):
    """Creates a transposed convolutional (deconvolution) layer.

    :param X
        Input tensor
    :param n_ch_in
        Number of input channels
    :param n_ch_out
        Number of output channels
    :param kernel_size
        Size of square shaped deconvolutional kernel
    :param strides
        Stride information
    """

    # Construct output shape of the deconvolution
    new_h = X.get_shape().as_list()[1]*strides[1]
    new_w = X.get_shape().as_list()[2]*strides[2]
    output_shape = [X.get_shape().as_list()[0], new_h, new_w, n_ch_out]

    # Note the in and out channels reversed for deconv shape
    shape = [kernel_size, kernel_size, n_ch_out, n_ch_in]
    W = tf.get_variable(name='W',
                        shape=shape,
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
    h = tf.nn.conv2d_transpose(X,
                               output_shape=output_shape,
                               filter=W,
                               strides=strides,
                               padding="SAME")

    return h


def relu(X):
    """Performs relu on the tensor.

    :param X
        Input tensor
    """
    return tf.nn.relu(X, name='relu')


def scaled_tanh(X):
    """Performs tanh activation to ensure range of 0,255 on positive output.

    :param X
        Input tensor
    """
    scale = tf.constant(255.0)
    shift = tf.constant(255.0)
    half = tf.constant(2.0)
    # out = tf.mul(tf.tanh(X), scale)  # range of [-255, 255]
    out = (scale*tf.tanh(X) + shift) / half
    # out = tf.add(out, shift)  # range of [0, 2*255]
    # out = tf.div(out, half)  # range of [0, 255]
    return out


def inst_norm(inputs, epsilon=1e-3, suffix=''):
    """
    Assuming TxHxWxC dimensions on the tensor, will normalize over
    the H,W dimensions. Use this before the activation layer.
    This function borrows from:
        http://r2rt.com/implementing-batch-normalization-in-tensorflow.html

    Note this is similar to batch_normalization, which normalizes each
    neuron by looking at its statistics over the batch.

    :param input_:
        input tensor of NHWC format
    """
    # Create scale + shift. Exclude batch dimension.
    stat_shape = inputs.get_shape().as_list()
    scale = tf.get_variable('INscale'+suffix,
                            initializer=tf.ones(stat_shape[3]))
    shift = tf.get_variable('INshift'+suffix,
                            initializer=tf.zeros(stat_shape[3]))

    inst_means, inst_vars = tf.nn.moments(inputs, axes=[1, 2],
                                          keep_dims=True)

    # Normalization
    inputs_normed = (inputs - inst_means) / tf.sqrt(inst_vars + epsilon)

    # Perform trainable shift.
    output = scale * inputs_normed + shift

    return output


def res_layer(X, n_ch, kernel_size, strides, style_labels, num_styles, norm_func):
    """Creates a residual block layer.

    :param X
        Input tensor
    :param n_ch
        Number of input channels
    :param kernel_size
        Size of square shaped convolutional kernel
    :param strides
        Stride information
    """
    h = conv2d(X, n_ch, n_ch, kernel_size, strides, name='W1', padding='VALID')
    h = relu(norm_func(h, style_labels, num_styles, suffix='1'))
    h = conv2d(h, n_ch, n_ch, kernel_size, strides, name='W2', padding='VALID')
    h = norm_func(h, style_labels, num_styles, suffix='2')

    # Crop for skip connection
    in_shape = X.get_shape().as_list()
    begin = [0, 2, 2, 0]
    size = [-1, in_shape[1]-4, in_shape[2]-4, -1]
    X_crop = tf.slice(X, begin=begin, size=size)

    # Residual skip connection
    # h = tf.add(h, X_crop, name='res_out')
    # print("residual : h")
    # print(h)
    # print("residual : X")
    # print(X)
    h = tf.add(h, X_crop, name='res_out')

    return h

def cond_inst_norm(inputs, labels, num_categories, var_epsilon=1e-5, suffix=''):
    # print("cond_inst_norm")
    with tf.variable_scope('CondInstNorm'):
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        if inputs_rank != 4:
            raise ValueError('Inputs %s is not a 4D tensor.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        params_shape = inputs_shape[-1:]
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined last dimension %s.' % (inputs.name, params_shape))

        # Allocate parameters for the beta and gamma of the normalization.
        cond_shape = tf.TensorShape([num_categories]).concatenate(params_shape)
        # print(cond_shape)
        bata_list = tf.get_variable('beta'+suffix, initializer=tf.zeros_initializer(), shape=cond_shape, dtype=dtype, trainable=True)
        beta = tf.gather(bata_list, labels)
        beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)

        gamma_list = tf.get_variable('gamma'+suffix, initializer=tf.ones_initializer(), shape=cond_shape, dtype=dtype, trainable=True)
        gamma = tf.gather(gamma_list, labels)
        gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)

        # Calculate the moments on the last axis (instance activations).
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)

        # Compute layer normalization using the batch_normalization function.
        inputs_normed = (inputs - mean) / tf.sqrt(variance + var_epsilon)

        # Perform trainable shift.
        outputs = gamma * inputs_normed + beta

        # outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, var_epsilon)
        outputs.set_shape(inputs_shape)

    return outputs


def weighted_inst_norm(inputs, weights, num_categories, var_epsilon=1e-5, suffix=''):
    # print("weight_inst_norm")
    with tf.variable_scope('CondInstNorm'):
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        if inputs_rank != 4:
            raise ValueError('Inputs %s is not a 4D tensor.' % inputs.name)
        dtype = inputs.dtype.base_dtype
        axis = [1, 2]
        params_shape = inputs_shape[-1:]
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined last dimension %s.' % (inputs.name, params_shape))

        # Allocate parameters for the beta and gamma of the normalization.
        weight_shape = tf.TensorShape([num_categories]).concatenate(params_shape)
        weights = tf.reshape(weights, weights.get_shape().concatenate([1] * params_shape.ndims))
        beta_list = tf.get_variable('beta'+suffix, initializer=tf.zeros_initializer(), shape=weight_shape, dtype=dtype, trainable=True)
        beta_weight = weights * beta_list
        beta_weight = tf.reduce_sum(beta_weight, 0, keep_dims=True)
        beta = tf.expand_dims(tf.expand_dims(beta_weight, 1), 1)

        gamma_list = tf.get_variable('gamma'+suffix, initializer=tf.ones_initializer(), shape=weight_shape, dtype=dtype, trainable=True)
        gamma_weight = weights * gamma_list
        gamma_weight = tf.reduce_sum(gamma_weight, 0, keep_dims=True)
        gamma = tf.expand_dims(tf.expand_dims(gamma_weight, 1), 1)

        # Calculate the moments on the last axis (instance activations).
        mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)
        # Compute layer normalization using the batch_normalization function.
        outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, var_epsilon)
        outputs.set_shape(inputs_shape)

    return outputs









