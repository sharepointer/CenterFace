import tensorflow as tf
from tensorflow.contrib import slim


def conv2d(x, num_ch, k_size=3, s_size=1, scope='conv_2d', DW=True):
    with tf.variable_scope(name_or_scope=scope):
        if DW:
            out = slim.separable_conv2d(x, num_outputs=num_ch, kernel_size=[k_size, k_size], stride=s_size)
        else:
            out = slim.conv2d(x, num_outputs=num_ch, kernel_size=[k_size, k_size], stride=s_size)
        return out


def downsample_2x(x, num_ch, scope='down2x'):
    with tf.variable_scope(name_or_scope=scope):
        out = slim.conv2d(x, num_ch, kernel_size=[3, 3], stride=2)
        return out


def downsample_4x(x, num_ch, scope='down4x'):
    with tf.variable_scope(name_or_scope=scope):
        out = slim.conv2d(x, num_ch, kernel_size=[3, 3], stride=2)
        out = slim.conv2d(out, num_ch, kernel_size=[3, 3], stride=2)
        return out


def downsample_8x(x, num_ch, scope='down8x'):
    with tf.variable_scope(name_or_scope=scope):
        out = slim.conv2d(x, num_ch, kernel_size=[3, 3], stride=2)
        out = slim.conv2d(out, num_ch, kernel_size=[3, 3], stride=2)
        out = slim.conv2d(out, num_ch, kernel_size=[3, 3], stride=2)
        return out


def upsample_2x(x, num_ch, scope='up2x'):
    with tf.variable_scope(name_or_scope=scope):
        out = slim.conv2d_transpose(x, num_ch, kernel_size=[2, 2], stride=2)
        return out


def upsample_4x(x, num_ch, scope='up4x'):
    with tf.variable_scope(name_or_scope=scope):
        out = slim.conv2d_transpose(x, num_ch, kernel_size=[2, 2], stride=2)
        out = slim.conv2d_transpose(out, num_ch, kernel_size=[2, 2], stride=2)
        return out


def upsample_8x(x, num_ch, scope='up8x'):
    with tf.variable_scope(name_or_scope=scope):
        out = slim.conv2d_transpose(x, num_ch, kernel_size=[2, 2], stride=2)
        out = slim.conv2d_transpose(out, num_ch, kernel_size=[2, 2], stride=2)
        out = slim.conv2d_transpose(out, num_ch, kernel_size=[2, 2], stride=2)
        return out


def model(inputs, is_training=False):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.separable_conv2d],
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
            path_ch = 6
            path1_ch = path_ch * 1
            path2_ch = path1_ch * 2
            path3_ch = path2_ch * 2
            path4_ch = path3_ch * 2

            net = slim.conv2d(inputs, 16, kernel_size=3, stride=2, scope='Conv')

            path1_1 = downsample_2x(net, path1_ch, scope='path1_1')
            path1_2 = conv2d(path1_1, path1_ch, scope='path1_2')
            path1_3 = conv2d(path1_2, path1_ch, scope='path1_3')
            path1_3_down_2x = downsample_2x(path1_3, path2_ch, 'path1_3_down_2x')

            path1_4 = conv2d(path1_3, path1_ch, scope='path1_4')
            path1_5 = conv2d(path1_4, path1_ch, scope='path1_5')
            path1_6 = conv2d(path1_5, path1_ch, scope='path1_6')
            path1_6_down_2x = downsample_2x(path1_6, path2_ch, 'path1_6_down_2x')
            path1_6_down_4x = downsample_4x(path1_6, path3_ch, 'path1_6_down_4x')

            with tf.variable_scope(name_or_scope='path2_1'):
                path2_1 = tf.identity(path1_3_down_2x)

            path2_2 = conv2d(path2_1, path2_ch, scope='path2_2')
            path2_3 = conv2d(path2_2, path2_ch, scope='path2_3')
            path2_3_down_2x = downsample_2x(path2_3, path3_ch, scope='path2_3_down_2x')
            path2_3_up_2x = upsample_2x(path2_3, path1_ch, scope='path2_3_up_2x')

            with tf.variable_scope(name_or_scope='path1_7'):
                path1_7 = conv2d(path1_6, path1_ch)
                path1_7 = tf.add(path1_7, path2_3_up_2x)

            path1_8 = conv2d(path1_7, path1_ch, scope='path1_8')
            path1_9 = conv2d(path1_8, path1_ch, scope='path1_9')
            path1_9_down_2x = downsample_2x(path1_9, path2_ch, 'path1_9_down_2x')
            path1_9_down_4x = downsample_4x(path1_9, path3_ch, 'path1_9_down_4x')
            path1_9_down_8x = downsample_8x(path1_9, path4_ch, 'path1_9_down_8x')

            with tf.variable_scope(name_or_scope='path2_4'):
                path2_4 = conv2d(path2_3, path2_ch)
                path2_4 = tf.add(path2_4, path1_6_down_2x)

            path2_5 = conv2d(path2_4, path2_ch, scope='path2_5')
            path2_6 = conv2d(path2_5, path2_ch, scope='path2_6')
            path2_6_down_2x = downsample_2x(path2_6, path3_ch, scope='path2_6_down_2x')
            path2_6_down_4x = downsample_4x(path2_6, path4_ch, scope='path2_6_down_4x')
            path2_6_up_2x = upsample_2x(path2_6, path1_ch, scope='path2_6_up_2x')

            with tf.variable_scope(name_or_scope='path3_1'):
                path3_1 = tf.add(path2_3_down_2x, path1_6_down_4x, name='path3_1')

            path3_2 = conv2d(path3_1, path3_ch, scope='path3_2')
            path3_3 = conv2d(path3_2, path3_ch, scope='path3_3')
            path3_3_down_2x = downsample_2x(path3_3, path4_ch, scope='path3_3_down_2x')
            path3_3_up_2x = upsample_2x(path3_3, path2_ch, scope='path3_3_up_2x')
            path3_3_up_4x = upsample_4x(path3_3, path1_ch, scope='path3_3_up_4x')

            with tf.variable_scope(name_or_scope='path1_10'):
                path1_10 = conv2d(path1_9, path1_ch)
                path1_10 = tf.add(path1_10, path2_6_up_2x)
                path1_10 = tf.add(path1_10, path3_3_up_4x)

            path1_11 = conv2d(path1_10, path1_ch, scope='path1_11')
            path1_12 = conv2d(path1_11, path1_ch, scope='path1_12')

            with tf.variable_scope(name_or_scope='path2_7'):
                path2_7 = conv2d(path2_6, path2_ch)
                path2_7 = tf.add(path2_7, path3_3_up_2x)
                path2_7 = tf.add(path2_7, path1_9_down_2x)

            path2_8 = conv2d(path2_7, path2_ch, scope='path2_8')
            path2_9 = conv2d(path2_8, path2_ch, scope='path2_9')

            with tf.variable_scope(name_or_scope='path3_4'):
                path3_4 = conv2d(path3_3, path3_ch)
                path3_4 = tf.add(path3_4, path1_9_down_4x)
                path3_4 = tf.add(path3_4, path2_6_down_2x)

            path3_5 = conv2d(path3_4, path3_ch, scope='path3_5')
            path3_6 = conv2d(path3_5, path3_ch, scope='path3_6')

            with tf.variable_scope(name_or_scope='path4_1'):
                path4_1 = tf.add(path3_3_down_2x, path2_6_down_4x)
                path4_1 = tf.add(path4_1, path1_9_down_8x, name='path4_1')

            path4_2 = conv2d(path4_1, path4_ch, scope='path4_2')
            path4_3 = conv2d(path4_2, path4_ch, scope='path4_3')

            path4_3_up_8x = upsample_8x(path4_3, path1_ch, scope='path4_3_up_8x')
            path3_6_up_4x = upsample_4x(path3_6, path1_ch, scope='path3_6_up_4x')
            path2_9_up_2x = upsample_2x(path2_9, path1_ch, scope='path2_9_up_2x')

            p3 = tf.concat([path1_12, path2_9_up_2x, path3_6_up_4x, path4_3_up_8x], axis=3, name='p3')
            p_out = slim.conv2d(p3, 24, kernel_size=[3, 3], stride=1, scope='p_out')

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
