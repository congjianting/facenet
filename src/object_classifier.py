#-*- coding: UTF-8 -*-
"""
See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# add nets and preprocessing method codes in slim package
import os
import math
import tensorflow as tf
import cv2
import time
import numpy as np
import tensorflow.contrib.slim as slim
import importlib

# sort top-5 for object prob
def process_top5_inds_prob( prob ):
    # top-5 index and prob
    prob = prob[0, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-prob), key=lambda x: x[1])]
    top_5_inds = []
    top_5_prob = []
    for i in range(min(5,len(prob))):
        top_5_inds.append(sorted_inds[i])
        top_5_prob.append(prob[sorted_inds[i]])

    return  top_5_inds,top_5_prob

class ObjectClassifier:

    # define inceptionv1_resent network
    def create_architecture_inceptionv1(self):
        # create net structor and size
        image_size        = 224
        # define input image placeholder
        self._image       = tf.placeholder(tf.uint8, shape=[224, 224, 3])

        # extend dim
        processed_images  = tf.expand_dims(tf.image.per_image_standardization(self._image), 0)
        self._phase_train = tf.placeholder(tf.bool)

        # create model
        network = importlib.import_module('models.inception_resnet_v1')

        # Build the inference graph
        prelogits, _ = network.inference(processed_images, 0.8,
                                         phase_train=self._phase_train, bottleneck_layer_size=128,
                                         weight_decay=0.00001)

        logits = slim.fully_connected(prelogits, self.num_classes, activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                weights_regularizer=slim.l2_regularizer(0.00001),
                scope='Logits', reuse=False)

        self._embeddings    = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # add softmax node add by cjt@20180224
        self._probabilities = tf.nn.softmax(logits, name='predicts')


    # ini method
    def __init__(self, net_name, model_path, num_classes=2):

        self.model_path   = model_path
        self.num_classes  = num_classes
        self._predictions = {}
        
        # set config
        tfconfig  = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth=True
        self.sess = tf.Session(config=tfconfig)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # check .meta exists
        if not os.path.isfile(self.model_path + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(self.model_path + '.meta'))

        # load network
        self.net_name = net_name
        with self.sess.as_default():
            # define user network
            self.create_architecture_inceptionv1()

            # load weights
            saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
            saver.restore(self.sess, self.model_path)

    # define object classify method
    def object_classify(self, image):

        # null
        if image is None:
            print("object_classify function input image not found")
            return []

        # predict object prob
        self._predictions["cls_prob"] = self.sess.run(self._probabilities, feed_dict={self._image: image, self._phase_train: False})

        # sort prob and index
        top_5_inds,top_5_prob = process_top5_inds_prob(self._predictions["cls_prob"])

        return top_5_inds[0], top_5_prob[0]

# test code
if __name__ == '__main__':

    print(tf.__version__)

    # # part1
    # # test resnet50v1 class
    #
    # # input parameters
    # # check point
    # checkpoints_dir   = '/Users/congjt/PycharmProjects/tfslim_predict/car_dirs_class8_model/resnet50/model.ckpt-120000'
    # # class_num
    # class_num_support = 8
    #
    # # input one image for test
    # image_path        = '../example/189e2b1473a111e79f66784f43a08a3cusrcol2017072822290102536.jpg'
    #
    # # create object-classifier instance
    # dir_classifier = ObjectClassifier('resnet_v1_50', checkpoints_dir, class_num_support)
    #
    # # read one image
    # im = cv2.imread(image_path)
    # # BGR->RGB
    # im = np.array(im)[:, :, ::-1]
    #
    # start_time = time.time()
    # # classify object
    # cls, prob = dir_classifier.object_classify(im)
    # end_time = time.time()
    #
    # # print result
    # print("resnet50v1 cls = %d, prob = %f" % (cls, prob))
    # print("resnet50v1 cost time: %f" % (end_time - start_time))
    #
    # # part2
    # # test mobilenet class
    #
    # # input parameters
    # # check point
    # checkpoints_dir   = '/Users/congjt/PycharmProjects/tfslim_predict/claim_class4_model/claim_V3/mobilenetv1/model.ckpt-60000'
    # # class_num
    # class_num_support = 4
    #
    # # input one image for test
    # image_path        = '../example/usrvinefc45a147c2011e7be94784f43a08a3c.jpg'
    #
    # # create object-classifier instance
    # claim_classifier = ObjectClassifier('MobilenetV1', checkpoints_dir, class_num_support)
    #
    # # read one image
    # im = cv2.imread(image_path)
    # # BGR->RGB
    # im = np.array(im)[:, :, ::-1]
    #
    # start_time = time.time()
    # # classify object
    # cls, prob = claim_classifier.object_classify(im)
    # end_time = time.time()
    #
    # # print result
    # print("MobilenetV1 cls = %d, prob = %f" % (cls, prob))
    # print("MobilenetV1 cost time: %f" % (end_time - start_time))
    #
    # # part3
    # test inceptionv1 class

    # input parameters
    # check point
    checkpoints_dir   = '../models/brands/20180224-151915/model-20180224-151915.ckpt-9000'
    # class_num
    class_num_support = 241

    # input one image for test
    image_path        = '../data/200种车款训练样本集/比亚迪-秦-A款/20150522065952986_蓝沪DZ1206.jpg'

    # create object-classifier instance
    brand_classifier = ObjectClassifier('InceptionV1', checkpoints_dir, class_num_support)

    # read one image
    im = cv2.imread(image_path)
    # BGR->RGB
    im = np.array(im)[:, :, ::-1]

    # resize to 224*224
    im_re = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)

    start_time = time.time()
    # classify object
    cls, prob = brand_classifier.object_classify(im_re)
    end_time = time.time()

    # print result
    print("inceptionv3 cls = %d, prob = %f" % (cls, prob))
    print("inceptionv3 cost time: %f"%(end_time-start_time))








