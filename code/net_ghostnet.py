import tensorflow as tf
import tf_slim as slim


def ghost_conv(x, ch_num, k_s=3, bn=slim.batch_norm, act=tf.nn.relu, name='ghost_conv'):
    with tf.variable_scope(name_or_scope=name):
        x1 = slim.conv2d(x, ch_num // 2, kernel_size=[k_s, k_s], activation_fn=None, normalizer_fn=None)
        x2 = slim.separable_conv2d(x1, None, [3, 3], activation_fn=None, normalizer_fn=None)
        y = tf.concat([x1, x2], axis=3)
        if bn:
            y = bn(y)
        if act:
            y = act(y)
        return y


def ghost_bottleneck(x, ch_exp, ch_out, s_s, name):
    with tf.variable_scope(name_or_scope=name):
        net = ghost_conv(x, ch_exp, k_s=1, name='expand')
        if 2 == s_s:
            net = slim.separable_conv2d(net, None, kernel_size=3, stride=s_s, normalizer_fn=slim.batch_norm, activation_fn=None, scope='depthwise')
        net = ghost_conv(net, ch_out, k_s=1, act=None, name='project')
        ch_in = int(x.get_shape().as_list()[3])
        if ch_in == ch_out and 2 != s_s:
            y = tf.add(net, x)
        else:
            y = net
    return y


def ghost_backbone(inputs, is_training=False):
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        normalizer_fn=slim.batch_norm,
                        activation_fn=tf.nn.relu,
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
            net = ghost_bottleneck(net, 32, 16, s_s=2, name='expanded_conv')
            feature_map_4 = net
            net = ghost_bottleneck(net, 72, 24, s_s=2, name='expanded_conv_1')
            net = ghost_bottleneck(net, 88, 24, s_s=1, name='expanded_conv_2')
            feature_map_8 = net
            net = ghost_bottleneck(net, 96, 40, s_s=2, name='expanded_conv_3')
            net = ghost_bottleneck(net, 240, 40, s_s=1, name='expanded_conv_4')
            net = ghost_bottleneck(net, 240, 40, s_s=1, name='expanded_conv_5')
            net = ghost_bottleneck(net, 120, 48, s_s=1, name='expanded_conv_6')
            net = ghost_bottleneck(net, 144, 48, s_s=1, name='expanded_conv_7')
            feature_map_16 = net
            net = ghost_bottleneck(net, 288, 96, s_s=2, name='expanded_conv_8')
            net = ghost_bottleneck(net, 576, 96, s_s=1, name='expanded_conv_9')
            net = ghost_bottleneck(net, 576, 96, s_s=1, name='expanded_conv_10')
            feature_map_32 = net

            return feature_map_4, feature_map_8, feature_map_16, feature_map_32
