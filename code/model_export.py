import tensorflow as tf
from tensorflow.python.framework import graph_util
import tensorflow.lite as tflite
from tensorflow.lite.python import lite_constants
import argparse
import os

import net_factory
from config import _C as CONFIG

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

args = argparse.ArgumentParser()
args.add_argument('--size', default=32, type=int)
args.add_argument('--tflite', default=0, type=int)
args.add_argument('--name', default='mobilenet_centerface', type=str)
ARGS = args.parse_args()

input_size = ARGS.size
export_lite = ARGS.tflite
export_name = ARGS.name

tf.reset_default_graph()

input_image = tf.placeholder(tf.float32, (1, input_size, input_size, 3), name='input_image')

if export_lite:
    is_quantize = False
    if export_lite & 2 == 2:
        is_quantize = True

    map_class, map_scale, map_offset, map_landmark = net_factory.build(input_image, backbone=CONFIG.MODEL.BACKBONE_NAME)
    # print(map_class)
    # print(map_scale)
    # print(map_offset)
    # print(map_landmark)

    map_class = tf.identity(map_class, name='pred_class_map')
    map_scale = tf.identity(map_scale, name='pred_scale_map')
    map_offset = tf.identity(map_offset, name='pred_offset_map')
    map_landmark = tf.identity(map_landmark, name='pred_landmark_map')

    # map_class_op = tf.identity(map_class, name='pred_class_map')
    # map_scale_op = tf.multiply(map_scale, 1.0, 'pred_scale_map')
    # map_offset_op = tf.multiply(map_offset, 1.0, 'pred_offset_map')
    # map_landmark_op = tf.multiply(map_landmark, 1.0, name='pred_landmark_map')

    if is_quantize:
        tf.contrib.quantize.create_eval_graph()

    saver = tf.train.Saver()

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    tflite_model_name = ''

    with tf.Session(config=gpu_config) as sess:
        new_checkpoint = tf.train.latest_checkpoint('./model_checkpoint')
        print('====== {}'.format(new_checkpoint))
        saver.restore(sess, new_checkpoint)

        # lite
        if is_quantize:
            tflite_model_name = 'centerface_tflite_quantize.tflite'
            # converter = tflite.TFLiteConverter.from_session(sess, [input_image], [map_class_op, map_scale_op, map_offset_op, map_landmark_op])
            converter = tflite.TFLiteConverter.from_session(sess, [input_image], [map_class, map_scale, map_offset, map_landmark])
            # converter.post_training_quantize = True
            converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
            input_arrays = converter.get_input_arrays()
            converter.quantized_input_stats = {input_arrays[0]: (127.5, 128.0)}
            tflite_model = converter.convert()
            open('./model_export/{}'.format(tflite_model_name), "wb").write(tflite_model)
        else:
            tflite_model_name = 'centerface_tflite_float.tflite'
            # converter = tflite.TFLiteConverter.from_session(sess, [input_image], [map_class_op, map_scale_op, map_offset_op, map_landmark_op])
            converter = tflite.TFLiteConverter.from_session(sess, [input_image], [map_class, map_scale, map_offset, map_landmark])
            tflite_model = converter.convert()
            open('./model_export/{}'.format(tflite_model_name), "wb").write(tflite_model)

    if is_quantize is False:
        os.chdir('model_export')
        os.system('./MNNConvert -f TFLITE --modelFile {} --MNNModel {}.mnn --bizCode mnn'.format(tflite_model_name, os.path.splitext(tflite_model_name)[0]))
        os.system('./mnn2bin {}.mnn'.format(os.path.splitext(tflite_model_name)[0]))
    else:
        os.chdir('model_export')
        os.system('./mnn2bin {}'.format(tflite_model_name))
else:
    map_class, map_scale, map_offset, map_landmark, _ = net_factory.build(input_image, backbone=CONFIG.MODEL.BACKBONE_NAME)
    # print(map_class)
    # print(map_scale)
    # print(map_offset)
    # print(map_landmark)
    map_class_op = tf.identity(map_class, name='pred_class_map')
    map_scale_op = tf.identity(map_scale, name='pred_scale_map')
    map_offset_op = tf.identity(map_offset, name='pred_offset_map')
    map_landmark_op = tf.identity(map_landmark, name='pred_landmark_map')
    # print('map_class_op:  {}'.format(map_class_op))
    # print('map_scale_op:  {}'.format(map_scale_op))
    # print('map_offset_op:  {}'.format(map_offset_op))
    # print('map_landmark_op:  {}'.format(map_landmark_op))
    saver = tf.train.Saver()
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=gpu_config) as sess:
        new_checkpoint = tf.train.latest_checkpoint('../checkpoints/checkpoints')
        print('{}'.format(new_checkpoint))
        saver.restore(sess, new_checkpoint)

        saver.save(sess, '../inference/{}.ckpt'.format(export_name))
        tf.train.write_graph(sess.graph_def, '../inference', '{}.pbtxt'.format(export_name))

        # freeze
        graph_def = tf.get_default_graph().as_graph_def()
        freeze_graph = graph_util.convert_variables_to_constants(sess, graph_def,
                                                                 ['pred_class_map',
                                                                  'pred_scale_map',
                                                                  'pred_offset_map',
                                                                  'pred_landmark_map'])
        open('../inference/{}.pb'.format(export_name), "wb").write(freeze_graph.SerializeToString())

    os.chdir('../inference')
    os.system('./MNNConvert -f TF --modelFile {}.pb --MNNModel {}.mnn --bizCode mnn'.format(export_name, export_name))
    os.system('./mnn2bin {}.mnn'.format(export_name))
