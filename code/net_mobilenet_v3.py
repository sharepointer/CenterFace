import tensorflow as tf
from tensorflow.contrib import slim


def bottleneck(x, ch_exp, ch_out, s_s, name):
    with tf.variable_scope(name_or_scope=name):
        if -1 != ch_exp:
            net = slim.conv2d(x, ch_exp, kernel_size=[1, 1], scope='expand')
        else:
            net = x

        net = slim.separable_conv2d(net, None, kernel_size=3, stride=s_s, scope='depthwise')
        net = slim.conv2d(net, ch_out, kernel_size=[1, 1], activation_fn=None, scope='project')

        ch_in = int(x.get_shape().as_list()[3])
        if ch_in == ch_out and 2 != s_s:
            y = tf.add(net, x)
        else:
            y = tf.identity(net)
    return y


def backbone(inputs, is_training=False):
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        normalizer_fn=slim.batch_norm,
                        activation_fn=tf.nn.relu6,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=None,
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='SAME'):
        with slim.arg_scope([slim.batch_norm],
                            center=True,
                            scale=True,
                            is_training=is_training):
            net = slim.conv2d(inputs, 16, kernel_size=3, stride=2, scope='Conv')
            feature_map_2 = net
            net = bottleneck(net, -1, 16, s_s=2, name='expanded_conv')
            feature_map_4 = net
            net = bottleneck(net, 72, 24, s_s=2, name='expanded_conv_1')
            net = bottleneck(net, 88, 24, s_s=1, name='expanded_conv_2')
            feature_map_8 = net
            net = bottleneck(net, 96, 40, s_s=2, name='expanded_conv_3')
            net = bottleneck(net, 240, 40, s_s=1, name='expanded_conv_4')
            net = bottleneck(net, 240, 40, s_s=1, name='expanded_conv_5')
            net = bottleneck(net, 120, 48, s_s=1, name='expanded_conv_6')
            net = bottleneck(net, 144, 48, s_s=1, name='expanded_conv_7')
            feature_map_16 = net
            net = bottleneck(net, 288, 96, s_s=2, name='expanded_conv_8')
            net = bottleneck(net, 576, 96, s_s=1, name='expanded_conv_9')
            net = bottleneck(net, 576, 96, s_s=1, name='expanded_conv_10')
            feature_map_32 = net

            return feature_map_4, feature_map_8, feature_map_16, feature_map_32
