import tensorflow as tf
import tf_slim as slim
from mobilenet import mobilenet_v2
from mobilenet import mobilenet_v3
import net_mobilenet_v3
import net_ghostnet


def upsample(X, num_ch, scope, upsample_bilinear=False):
    with tf.variable_scope(name_or_scope=scope):
        if upsample_bilinear:
            map_size = int(X.get_shape().as_list()[1] * 2)
            out = tf.image.resize_bilinear(X, [map_size, map_size])
            out = slim.conv2d(out, num_ch, kernel_size=[1, 1], stride=1)
        else:
            out = slim.conv2d_transpose(X, num_ch, kernel_size=[2, 2], stride=2)
        return out


def build(inputs, is_training=False, ssh_head=True, num_head_ch=24, upsample_bilinear=False, backbone='mbv2_1.0'):
    # feature_map_4, feature_map_8, feature_map_16, feature_map_32 = net_ghostnet.ghost_backbone(inputs, is_training)
    # feature_map_4, feature_map_8, feature_map_16, feature_map_32 = net_mobilenet_v3.backbone(inputs, is_training)

    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training)):
        _, endpoints = mobilenet_v2.mobilenet_base(inputs, conv_defs=mobilenet_v2.V2_DEF, depth_multiplier=0.5, finegrain_classification_mode=True)
    feature_map_4 = endpoints['layer_4']
    feature_map_8 = endpoints['layer_7']
    feature_map_16 = endpoints['layer_14']
    feature_map_32 = endpoints['layer_18']

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
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
            p6 = slim.conv2d(feature_map_32, num_head_ch, kernel_size=[1, 1], stride=1, scope='p6_conv_1x1')
            p6 = tf.identity(p6, 'p6')
            print(p6)

            p5_0 = slim.conv2d(feature_map_16, num_head_ch, kernel_size=[1, 1], stride=1, scope='p5_conv_1x1')
            p5_1 = upsample(p6, num_head_ch, upsample_bilinear=upsample_bilinear, scope='p5_upsample')
            p5 = tf.identity((p5_0 + p5_1), name='p5')
            print(p5)

            p4_0 = slim.conv2d(feature_map_8, num_head_ch, kernel_size=[1, 1], stride=1, scope='p4_conv_1x1')
            p4_1 = upsample(p5, num_head_ch, upsample_bilinear=upsample_bilinear, scope='p4_upsample')
            p4 = tf.identity((p4_0 + p4_1), name='p4')
            print(p4)

            p3_0 = slim.conv2d(feature_map_4, num_head_ch, kernel_size=[1, 1], stride=1, scope='p3_conv_1x1')
            p3_1 = upsample(p4, num_head_ch, upsample_bilinear=upsample_bilinear, scope='p3_upsample')
            p3 = tf.identity((p3_0 + p3_1), name='p3')
            print(p3)

            p_reg = slim.conv2d(p3, num_head_ch, kernel_size=[3, 3], stride=1)
            if ssh_head:
                p_out_1 = slim.conv2d(p3, num_head_ch // 2, kernel_size=[3, 3], stride=1)

                p_out_top = slim.conv2d(p3, num_head_ch // 4, kernel_size=[3, 3], stride=1)

                p_out_2 = slim.conv2d(p_out_top, num_head_ch // 4, kernel_size=[3, 3], stride=1)

                p_out_3 = slim.conv2d(p_out_top, num_head_ch // 4, kernel_size=[3, 3], stride=1)
                p_out_3 = slim.conv2d(p_out_3, num_head_ch // 4, kernel_size=[3, 3], stride=1)

                p_class = tf.concat([p_out_1, p_out_2, p_out_3], axis=3)
            else:
                p_class = slim.conv2d(p3, num_head_ch, kernel_size=[3, 3], stride=1)

            map_class = slim.conv2d(p_class, 1, kernel_size=1, stride=1,
                                    normalizer_fn=None, activation_fn=tf.nn.sigmoid, biases_initializer=tf.zeros_initializer(),
                                    scope='pred_class')
            print(map_class)

            map_scale = slim.conv2d(p_reg, 2, kernel_size=1, stride=1,
                                    normalizer_fn=None, activation_fn=None, biases_initializer=tf.zeros_initializer(),
                                    scope='pred_scale')
            print(map_scale)

            map_offset = slim.conv2d(p_reg, 2, kernel_size=1, stride=1,
                                     normalizer_fn=None, activation_fn=None, biases_initializer=tf.zeros_initializer(),
                                     scope='pred_offset')
            print(map_offset)

            map_landmark = slim.conv2d(p_reg, 10, kernel_size=1, stride=1,
                                       normalizer_fn=None, activation_fn=None, biases_initializer=tf.zeros_initializer(),
                                       scope='pred_landmark')
            print(map_landmark)

            map_semantic = slim.conv2d(p_reg, 2, kernel_size=1, stride=1,
                                       normalizer_fn=None, activation_fn=None, biases_initializer=tf.zeros_initializer(),
                                       scope='pred_semantic')
            print(map_semantic)

            if is_training:
                return map_class, map_scale, map_offset, map_landmark, map_semantic
            else:
                feat_max_pool = tf.nn.max_pool2d(map_class, ksize=3, strides=1, padding='SAME')
                feat_no_face = tf.zeros_like(map_class)
                map_class = tf.where(tf.equal(feat_max_pool, map_class), map_class, feat_no_face, name='pred_class')
                return map_class, map_scale, map_offset, map_landmark
