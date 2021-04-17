import tensorflow as tf
from tensorflow.contrib import slim


def oct_norm_and_act(x, bn=slim.batch_norm, act=tf.nn.relu):
    if bn:
        x = bn(x)
    if act:
        x = act(x)
    return x


def oct_upsample(x):
    size_ = int(x.get_shape().as_list()[1] * 2)
    print('============', x,size_)
    return tf.image.resize_nearest_neighbor(x, [size_, size_])


def oct_conv2d_first(x, num_output, k_s=1, s_s=1, bn=slim.batch_norm, act=tf.nn.relu, alpha=0.6, name='oct_conv2d'):
    with tf.variable_scope(name_or_scope=name):
        if 2 == s_s:
            x = slim.avg_pool2d(x, kernel_size=2, stride=2, padding='SAME')

        num_h = int(num_output * alpha)
        num_l = num_output - num_h

        net_h = slim.conv2d(x, num_h, kernel_size=[k_s, k_s], activation_fn=None, normalizer_fn=None)
        net_l_plool = slim.avg_pool2d(x, kernel_size=2, stride=2, padding='SAME')
        net_l = slim.conv2d(net_l_plool, num_l, kernel_size=[k_s, k_s], activation_fn=None, normalizer_fn=None)

        net_h = oct_norm_and_act(net_h, bn=bn, act=act)
        net_l = oct_norm_and_act(net_l, bn=bn, act=act)

        return net_h, net_l


def oct_depthwise2d(x, k_s=3, s_s=1, bn=slim.batch_norm, act=tf.nn.relu, name='oct_depthwise2d'):
    with tf.variable_scope(name_or_scope=name):
        net_h, net_l = x
        net_h = slim.separable_conv2d(net_h, None, kernel_size=k_s, stride=s_s, normalizer_fn=None, activation_fn=None)
        net_l = slim.separable_conv2d(net_l, None, kernel_size=k_s, stride=s_s, normalizer_fn=None, activation_fn=None)

        net_h = oct_norm_and_act(net_h, bn=bn, act=act)
        net_l = oct_norm_and_act(net_l, bn=bn, act=act)

        return net_h, net_l


def oct_conv2d(x, num_output, k_s=1, bn=slim.batch_norm, act=tf.nn.relu, alpha=0.6, name='oct_conv2d'):
    with tf.variable_scope(name_or_scope=name):
        num_h = int(num_output * alpha)
        num_l = num_output - num_h
        net_h, net_l = x

        net_h2h = slim.conv2d(net_h, num_h, kernel_size=[k_s, k_s], activation_fn=None, normalizer_fn=None)
        net_l2l = slim.conv2d(net_l, num_l, kernel_size=[k_s, k_s], activation_fn=None, normalizer_fn=None)

        net_h2l = slim.avg_pool2d(net_h, kernel_size=2, stride=2, padding='SAME')
        net_h2l = slim.conv2d(net_h2l, num_l, kernel_size=[k_s, k_s], activation_fn=None, normalizer_fn=None)

        net_l2h = slim.conv2d(net_l, num_h, kernel_size=[k_s, k_s], activation_fn=None, normalizer_fn=None)
        net_l2h = oct_upsample(net_l2h)

        net_h = net_h2h + net_l2h
        net_l = net_l2l + net_h2l

        net_h = oct_norm_and_act(net_h, bn=bn, act=act)
        net_l = oct_norm_and_act(net_l, bn=bn, act=act)

        return net_h, net_l


def oct_conv2d_last(x, num_output, k_s=1, bn=slim.batch_norm, act=tf.nn.relu, name='oct_conv2d'):
    with tf.variable_scope(name_or_scope=name):
        net_h, net_l = x

        net_h2h = slim.conv2d(net_h, num_output, kernel_size=[k_s, k_s], activation_fn=None, normalizer_fn=None)
        net_l2l = slim.conv2d(net_l, num_output, kernel_size=[k_s, k_s], activation_fn=None, normalizer_fn=None)
        net_l2l = oct_upsample(net_l2l)

        net = tf.add(net_h2h, net_l2l)
        net = oct_norm_and_act(net, bn=bn, act=act)
        return net


def oct_bottleneck(x, ch_exp, ch_out, s_s, name):
    with tf.variable_scope(name_or_scope=name):
        net_h_in, net_l_in = x
        net = oct_conv2d(x, ch_exp, k_s=1, name='expand')
        net = oct_depthwise2d(net, s_s=s_s, name='depthwise')
        net = oct_conv2d(net, ch_out, k_s=1, name='project')
        net_h_out, net_l_out = net

        ch_in = int(net_h_in.get_shape().as_list()[3])
        ch_out = int(net_h_out.get_shape().as_list()[3])
        if ch_in == ch_out and 2 != s_s:
            net_h = tf.add(net_h_out, net_h_in)
        else:
            net_h = net_h_out

        ch_in = int(net_l_in.get_shape().as_list()[3])
        ch_out = int(net_l_out.get_shape().as_list()[3])
        if ch_in == ch_out and 2 != s_s:
            net_l = tf.add(net_l_out, net_l_in)
        else:
            net_l = net_l_out

    return net_h, net_l


def otc_backbone(inputs, is_training=False):
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
            # net = slim.conv2d(inputs, 16, kernel_size=3, stride=2, scope='Conv')
            net = oct_conv2d_first(inputs, 16, k_s=3, s_s=2, name='Conv')
            net = oct_bottleneck(net, 32, 16, s_s=2, name='expanded_conv')
            feature_map_4 = net
            net = oct_bottleneck(net, 72, 24, s_s=2, name='expanded_conv_1')
            net = oct_bottleneck(net, 88, 24, s_s=1, name='expanded_conv_2')
            feature_map_8 = net
            net = oct_bottleneck(net, 96, 40, s_s=2, name='expanded_conv_3')
            net = oct_bottleneck(net, 240, 40, s_s=1, name='expanded_conv_4')
            net = oct_bottleneck(net, 240, 40, s_s=1, name='expanded_conv_5')
            net = oct_bottleneck(net, 120, 48, s_s=1, name='expanded_conv_6')
            net = oct_bottleneck(net, 144, 48, s_s=1, name='expanded_conv_7')
            feature_map_16 = net
            net = oct_bottleneck(net, 288, 96, s_s=2, name='expanded_conv_8')
            net = oct_bottleneck(net, 576, 96, s_s=1, name='expanded_conv_9')
            net = oct_bottleneck(net, 576, 96, s_s=1, name='expanded_conv_10')
            feature_map_32 = net
            print(feature_map_4)
            print(feature_map_8)
            print(feature_map_16)
            print(feature_map_32)

            return feature_map_4, feature_map_8, feature_map_16, feature_map_32


def upsample(X, num_ch, scope, mode='D'):
    with tf.variable_scope(name_or_scope=scope):
        if 'B' == mode:
            map_size = int(X.get_shape().as_list()[1] * 2)
            out = tf.image.resize_bilinear(X, [map_size, map_size], name='bilinear')
            out = slim.conv2d(out, num_ch, kernel_size=[1, 1], stride=1, scope='conv_1x1')
        else:
            out = slim.conv2d_transpose(X, num_ch, kernel_size=[2, 2], stride=2, scope='deconv_2x2')

        return out


def model(inputs, is_training=False, ssh_head=True, num_head_ch=24):
    feature_map_4, feature_map_8, feature_map_16, feature_map_32 = otc_backbone(inputs, is_training=is_training)

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
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
            # p6 = slim.conv2d(feature_map_32, num_head_ch, kernel_size=[1, 1], stride=1, scope='p6_conv_1x1')
            p6 = oct_conv2d_last(feature_map_32, num_head_ch, name='p6_conv_1x1')
            p6 = tf.identity(p6, 'p6')
            print(p6)

            # p5_0 = slim.conv2d(feature_map_16, num_head_ch, kernel_size=[1, 1], stride=1, scope='p5_conv_1x1')
            p5_0 = oct_conv2d_last(feature_map_16, num_head_ch, name='p5_conv_1x1')
            p5_1 = upsample(p6, num_head_ch, scope='p5_upsample')
            p5 = tf.identity((p5_0 + p5_1), name='p5')
            print(p5)

            # p4_0 = slim.conv2d(feature_map_8, num_head_ch, kernel_size=[1, 1], stride=1, scope='p4_conv_1x1')
            p4_0 = oct_conv2d_last(feature_map_8, num_head_ch, name='p4_conv_1x1')
            p4_1 = upsample(p5, num_head_ch, scope='p4_upsample')
            p4 = tf.identity((p4_0 + p4_1), name='p4')
            print(p4)

            # p3_0 = slim.conv2d(feature_map_4, num_head_ch, kernel_size=[1, 1], stride=1, scope='p3_conv_1x1')
            p3_0 = oct_conv2d_last(feature_map_4, num_head_ch, name='p3_conv_1x1')
            p3_1 = upsample(p4, num_head_ch, scope='p3_upsample')
            p3 = tf.identity((p3_0 + p3_1), name='p3')
            print(p3)

            if ssh_head:
                p_out_1 = slim.conv2d(p3, num_head_ch // 2, kernel_size=[3, 3], stride=1)

                p_out_top = slim.conv2d(p3, num_head_ch // 4, kernel_size=[3, 3], stride=1)

                p_out_2 = slim.conv2d(p_out_top, num_head_ch // 4, kernel_size=[3, 3], stride=1)

                p_out_3 = slim.conv2d(p_out_top, num_head_ch // 4, kernel_size=[3, 3], stride=1)
                p_out_3 = slim.conv2d(p_out_3, num_head_ch // 4, kernel_size=[3, 3], stride=1)

                p_out = tf.concat([p_out_1, p_out_2, p_out_3], axis=3)
            else:
                p_out = slim.conv2d(p3, num_head_ch, kernel_size=[3, 3], stride=1)

            p_out = tf.identity(p_out, name='p_out')
            print(p_out)

            map_class = slim.conv2d(p_out, 1, kernel_size=[1, 1], stride=1,
                                    normalizer_fn=None, activation_fn=tf.nn.sigmoid, biases_initializer=tf.zeros_initializer(),
                                    scope='pred_class')
            print(map_class)

            map_scale = slim.conv2d(p_out, 2, kernel_size=[1, 1], stride=1,
                                    normalizer_fn=None, activation_fn=None, biases_initializer=tf.zeros_initializer(),
                                    scope='pred_scale')
            print(map_scale)

            map_offset = slim.conv2d(p_out, 2, kernel_size=[1, 1], stride=1,
                                     normalizer_fn=None, activation_fn=None, biases_initializer=tf.zeros_initializer(),
                                     scope='pred_offset')
            print(map_offset)

            map_landmark = slim.conv2d(p_out, 10, kernel_size=[1, 1], stride=1,
                                       normalizer_fn=None, activation_fn=None, biases_initializer=tf.zeros_initializer(),
                                       scope='pred_landmark')
            print(map_landmark)

    return map_class, map_scale, map_offset, map_landmark
