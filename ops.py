import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.001)
weight_regularizer = None

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv', reuse=False):
    with tf.variable_scope(scope):
        if pad > 0 :
            if (kernel - stride) % 2 == 0:
                pad_top = pad
                pad_bottom = pad
                pad_left = pad
                pad_right = pad

            else:
                pad_top = pad
                pad_bottom = kernel - stride - pad_top
                pad_left = pad
                pad_right = kernel - stride - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding='VALID', reuse=reuse)
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias, reuse=reuse)
        return x

def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        else:
            output_shape =[x_shape[0], x_shape[1] * stride + max(kernel - stride, 0), x_shape[2] * stride + max(kernel - stride, 0), channels]

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x

def fully_conneted(x, channels, use_bias=True, sn=False, scope='fully',reuse=False):
    with tf.variable_scope(scope):
        # x = tf.layers.flatten(x)
        # shape = x.get_shape().as_list()
        # x = tf.reshape(x, [shape[0], shape[1]*shape[2]*shape[3]])
        shape = x.get_shape().as_list()
        x_channel = shape[-1]

        if sn :
            w = tf.get_variable("kernel", [x_channel, channels], tf.float32, initializer=weight_init, regularizer=weight_regularizer)
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else :
                x = tf.matmul(x, spectral_norm(w))

        else :
            x = tf.layers.dense(x, units=channels, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias, reuse=reuse)
        return x



def gaussian_noise_layer(x, is_training=False):
    if is_training :
        noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
        return x + noise
    else :
        return x

##################################################################################
# Block
##################################################################################

def resblock(x_init, channels, use_bias=True, sn=False, scope='resblock', reuse=False):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn, reuse=reuse)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn, reuse=reuse)
            x = instance_norm(x)

        return x + x_init

def basic_block(x_init, channels, use_bias=True, sn=False, scope='basic_block') :
    with tf.variable_scope(scope) :
        x = lrelu(x_init, 0.2)
        x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        x = lrelu(x, 0.2)
        x = conv_avg(x, channels, use_bias=use_bias, sn=sn)

        shortcut = avg_conv(x_init, channels, use_bias=use_bias, sn=sn)

        return x + shortcut

def mis_resblock(x_init, z, channels, use_bias=True, sn=False, scope='mis_resblock') :
    with tf.variable_scope(scope) :
        z = tf.reshape(z, shape=[-1, 1, 1, z.shape[-1]])
        z = tf.tile(z, multiples=[1, x_init.shape[1], x_init.shape[2], 1]) # expand

        with tf.variable_scope('mis1') :
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn, scope='conv3x3')
            x = instance_norm(x)

            x = tf.concat([x, z], axis=-1)
            x = conv(x, channels * 2, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_0')
            x = relu(x)

            x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_1')
            x = relu(x)

        with tf.variable_scope('mis2') :
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn, scope='conv3x3')
            x = instance_norm(x)

            x = tf.concat([x, z], axis=-1)
            x = conv(x, channels * 2, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_0')
            x = relu(x)

            x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='conv1x1_1')
            x = relu(x)

        return x + x_init

def avg_conv(x, channels, use_bias=True, sn=False, scope='avg_conv') :
    with tf.variable_scope(scope) :
        x = avg_pooling(x, kernel=2, stride=2)
        x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn)

        return x

def conv_avg(x, channels, use_bias=True, sn=False, scope='conv_avg') :
    with tf.variable_scope(scope) :
        x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
        x = avg_pooling(x, kernel=2, stride=2)

        return x

def expand_concat(x, z) :
    shape_x = x.get_shape().as_list()
    shape_z= z.get_shape().as_list()
    z = tf.reshape(z, shape=[shape_z[0], 1, 1, -1])
    z = tf.tile(z, multiples=[1, shape_x[1], shape_x[2], 1])  # expand
    x = tf.concat([x, z], axis=-1)

    return x

##################################################################################
# Sampling
##################################################################################

def down_sample(x) :
    return avg_pooling(x, kernel=3, stride=2, pad=1)

def avg_pooling(x, kernel=2, stride=2, pad=0) :
    if pad > 0 :
        if (kernel - stride) % 2 == 0:
            pad_top = pad
            pad_bottom = pad
            pad_left = pad
            pad_right = pad

        else:
            pad_top = pad
            pad_bottom = kernel - stride - pad_top
            pad_left = pad
            pad_right = kernel - stride - pad_left

        x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

    return tf.layers.average_pooling2d(x, pool_size=kernel, strides=stride, padding='VALID')

def global_avg_pooling(x):
    # gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    gap = tf.reduce_mean(x, axis=[1, 2])
    gap = tf.expand_dims(tf.expand_dims(gap, axis=1), axis=1)
    return gap

def z_sample(mean, logvar) :
    eps = tf.random_normal(shape=tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)

    return mean + tf.exp(logvar * 0.5) * eps

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.maximum(x, alpha * x)  #tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    # return tf.contrib.layers.instance_norm(x,
    #                                        epsilon=1e-05,
    #                                        center=True, scale=True,
    #                                        scope=scope)
    shape = x.get_shape().as_list()
    depth = shape[3]
    with tf.variable_scope(scope):
        scale = tf.get_variable("scale", shape=[depth], initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", shape=[depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv
    return scale * normalized + offset



def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x, center=True, scale=True, scope=scope)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)


    return w_norm

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(type, real, fake, fake_random=None, content=False):
    n_scale = len(real)
    loss = []

    real_loss = 0
    fake_loss = 0
    fake_random_loss = 0

    if content :
        for i in range(n_scale):
            if type == 'lsgan' :
                # real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
                # fake_loss = tf.reduce_mean(tf.square(fake[i]))
                real_loss=tf.reduce_mean(tf.square(real[i]-tf.random_uniform(shape=real[i].shape,minval=0.9,maxval=1.1)))
                fake_loss=tf.reduce_mean(tf.square(fake[i]-tf.random_uniform(shape=real[i].shape,minval=0,maxval=0.2)))

            if type =='gan' :
                real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
                fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i]), logits=fake[i]))

            loss.append(real_loss + fake_loss)

    else :
        for i in range(n_scale) :
            if type == 'lsgan' :
                real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
                fake_loss = tf.reduce_mean(tf.square(fake[i]))
                # fake_random_loss = tf.reduce_mean(tf.square(fake_random[i]))

            if type == 'gan' :
                real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
                fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i]), logits=fake[i]))
                # fake_random_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_random[i]), logits=fake_random[i]))

            loss.append(real_loss * 2 + fake_loss) #+ fake_random_loss)

    return sum(loss)


def generator_loss(type, fake, content=False):
    n_scale = len(fake)
    loss = []

    fake_loss = 0

    if content :
        for i in range(n_scale):
            if type =='lsgan' :
                fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 0.5))

            if type == 'gan' :
                fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=0.5 * tf.ones_like(fake[i]), logits=fake[i]))

            loss.append(fake_loss)
    else :
        for i in range(n_scale) :
            if type == 'lsgan' :
                fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 1.0))

            if type == 'gan' :
                fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake[i]), logits=fake[i]))

            loss.append(fake_loss)


    return sum(loss)


def l2_regularize(x) :
    loss = tf.reduce_mean(tf.square(x))

    return loss

def kl_loss(mu, logvar) :
    loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(logvar) - 1 - logvar, axis=-1)
    loss = tf.reduce_mean(loss)
    return loss

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


# def SSIM_LOSS(img1, img2, size=11, sigma=1.5):
#     window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
#     K1 = 0.01
#     K2 = 0.03
#     L = 1  # depth of image (255 in case the image has a differnt scale)
#     C1 = (K1*L)**2
#     C2 = (K2*L)**2
#     mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
#     mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
#     mu1_sq = mu1*mu1
#     mu2_sq = mu2*mu2
#     mu1_mu2 = mu1*mu2
#     sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
#     sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
#     sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
#
#     value = (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
#     value = tf.reduce_mean(value)
#     return value

def SSIM_LOSS(img1, img2, size = 11, sigma = 1.5):
    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    k1 = 0.01
    k2 = 0.03
    L = 1  # depth of image (255 in case the image has a different scale)
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
    mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
    sigma1_2 = tf.nn.conv2d(img1 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2

    # value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    value = tf.reduce_mean(ssim_map, axis = [0, 1, 2, 3])
    return 1-value


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis = -1)
    x_data = np.expand_dims(x_data, axis = -1)

    y_data = np.expand_dims(y_data, axis = -1)
    y_data = np.expand_dims(y_data, axis = -1)

    x = tf.constant(x_data, dtype = tf.float32)
    y = tf.constant(y_data, dtype = tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)


def RGB_SSIM_LOSS(img1, img2):
    loss1 = SSIM_LOSS(tf.expand_dims(img1[:, :, :, 0], axis=-1), tf.expand_dims(img2[:, :, :, 0], axis=-1))
    loss2 = SSIM_LOSS(tf.expand_dims(img1[:, :, :, 1], axis=-1), tf.expand_dims(img2[:, :, :, 1], axis=-1))
    loss3 = SSIM_LOSS(tf.expand_dims(img1[:, :, :, 2], axis=-1), tf.expand_dims(img2[:, :, :, 2], axis=-1))
    return (loss1 + loss2 + loss3) / 3

def MG(img):
    kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
    kernel = tf.expand_dims(kernel, axis=-1)
    kernel = tf.expand_dims(kernel, axis=-1)
    grads = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
    grads = tf.reduce_mean(tf.abs(grads), axis=[1,2,3])
    return grads

def SD(img):
    u=tf.reduce_mean(img, axis=[1,2,3])
    square=tf.reduce_mean(tf.square(img-u), axis=[1,2,3])
    sd=tf.sqrt(square)
    return sd

def L1_norm(source_en_a, source_en_b):
    result = []
    narry_a = source_en_a
    narry_b = source_en_b

    dimension = source_en_a.shape

    # caculate L1-norm
    temp_abs_a = tf.abs(narry_a)
    temp_abs_b = tf.abs(narry_b)
    _l1_a = tf.reduce_sum(temp_abs_a,3)
    _l1_b = tf.reduce_sum(temp_abs_b,3)

    _l1_a = tf.reduce_sum(_l1_a, 0)
    _l1_b = tf.reduce_sum(_l1_b, 0)
    l1_a = _l1_a.eval()
    l1_b = _l1_b.eval()

    # caculate the map for source images
    mask_value = l1_a + l1_b

    mask_sign_a = l1_a/mask_value
    mask_sign_b = l1_b/mask_value

    array_MASK_a = mask_sign_a
    array_MASK_b = mask_sign_b

    for i in range(dimension[3]):
        temp_matrix = array_MASK_a*narry_a[0,:,:,i] + array_MASK_b*narry_b[0,:,:,i]
        result.append(temp_matrix)

    result = np.stack(result, axis=-1)

    resule_tf = np.reshape(result, (dimension[0], dimension[1], dimension[2], dimension[3]))

    return resule_tf