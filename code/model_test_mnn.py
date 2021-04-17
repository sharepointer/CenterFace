from __future__ import print_function
import MNN
import numpy as np
import cv2


def inference():
    """ inference mobilenet_v1 using a specific picture """
    # interpreter = MNN.Interpreter("./model_export/centerface_tf_mobilenetV3_small_q.mnn")
    interpreter = MNN.Interpreter("./model_export/centerface_tf_mobilenetV3_small.mnn")
    # interpreter.setCacheFile('.tempcache')
    config = {}
    config['precision'] = 'low'
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    image = cv2.imread('./model_export/images/img1.jpg')
    # cv2 read as bgr format
    # image = image[..., ::-1]
    # change to rgb format
    # image = cv2.resize(image, (224, 224))
    # resize to mobile_net tensor size
    # image = image - (103.94, 116.78, 123.68)
    # image = image * (0.017, 0.017, 0.017)
    image = image - (127.5, 127.5, 127.58)
    image = image * (0.00784314, 0.00784314, 0.00784314)
    # preprocess it
    image = image.transpose((2, 0, 1))
    # change numpy data type as np.float32 to match tensor's format
    image = image.astype(np.float32)
    # cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    tmp_input = MNN.Tensor((1, 3, 320, 320), MNN.Halide_Type_Float, \
                           image, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    print(output_tensor)
    # #constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    # tmp_output = MNN.Tensor((1, 1001), MNN.Halide_Type_Float, np.ones([1, 1001]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
    # output_tensor.copyToHostTensor(tmp_output)
    # print("expect 983")
    # print("output belong to class: {}".format(np.argmax(tmp_output.getData())))


if __name__ == "__main__":
    inference()
