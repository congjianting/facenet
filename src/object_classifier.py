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

from nets          import nets_factory
from nets          import resnet_v1


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

default_image_size     = 224
default_embedding_size = 128

class ObjectClassifier:

    # define resnet_v1_50 network
    def create_architecture_resnet_v1_50(self):

        # define input image placeholder
        self._image = tf.placeholder(tf.uint8, shape=[None, None, 3])

        # resize
        self._imagere = tf.image.resize_images(self._image, (self._image_size, self._image_size))

        # extend dim
        processed_images  = tf.expand_dims(tf.image.per_image_standardization(self._imagere), 0)

        network_fn = nets_factory.get_network_fn(
                                                'resnet_v1_50',
                                                num_classes=None,
                                                weight_decay=0.00004,
                                                is_training=False)

        # Build the inference graph
        logits_tmp, _ = network_fn(processed_images)

        # create embeddings
        prelogits = slim.fully_connected(logits_tmp, self._embeddings_size, activation_fn=None, scope='Bottleneck', reuse=False)

        logits = slim.fully_connected(prelogits, self.num_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(0.00001),
                                      scope='Logits', reuse=False)

        self._embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # add softmax node add by cjt@20180224
        self._probabilities = tf.nn.softmax(logits, name='predicts')

    # define inceptionv1_resent network
    def create_architecture_inceptionv1(self):

        # define input image placeholder
        self._image = tf.placeholder(tf.uint8, shape=[None, None, 3])

        # resize
        self._imagere = tf.image.resize_images(self._image, (self._image_size, self._image_size))

        # extend dim
        processed_images  = tf.expand_dims(tf.image.per_image_standardization(self._imagere), 0)

        # create model
        network = importlib.import_module('models.inception_resnet_v1')

        # Build the inference graph
        prelogits, _ = network.inference(processed_images, 0.8,
                                         phase_train=False, bottleneck_layer_size=self._embeddings_size,
                                         weight_decay=0.00001)

        logits = slim.fully_connected(prelogits, self.num_classes, activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                weights_regularizer=slim.l2_regularizer(0.00001),
                scope='Logits', reuse=False)

        self._embeddings    = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # add softmax node add by cjt@20180224
        self._probabilities = tf.nn.softmax(logits, name='predicts')

    # define resnet_v2_50 network
    def create_architecture_resnet_v2_50(self):

        # define input image placeholder
        self._image = tf.placeholder(tf.uint8, shape=[None, None, 3])

        # resize
        self._imagere = tf.image.resize_images(self._image, (self._image_size, self._image_size))

        # extend dim
        processed_images  = tf.expand_dims(tf.image.per_image_standardization(self._imagere), 0)

        network_fn = nets_factory.get_network_fn(
                                                'resnet_v2_50',
                                                num_classes=None,
                                                weight_decay=0.00004,
                                                is_training=False)

        # Build the inference graph
        logits_tmp, _ = network_fn(processed_images)

        # create embeddings
        prelogits = slim.fully_connected(logits_tmp, self._embeddings_size, activation_fn=None, scope='Bottleneck', reuse=False)

        logits = slim.fully_connected(prelogits, self.num_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(0.00001),
                                      scope='Logits', reuse=False)

        self._embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # add softmax node add by cjt@20180224
        self._probabilities = tf.nn.softmax(logits, name='predicts')

    # define inception_v2 network
    def create_architecture_inception_v2(self):

        # define input image placeholder
        self._image = tf.placeholder(tf.uint8, shape=[None, None, 3])

        # resize
        self._imagere = tf.image.resize_images(self._image, (self._image_size, self._image_size))

        # extend dim
        processed_images  = tf.expand_dims(tf.image.per_image_standardization(self._imagere), 0)

        network_fn = nets_factory.get_network_fn(
                                                'inception_v2',
                                                num_classes=None,
                                                weight_decay=0.00004,
                                                is_training=False)

        # Build the inference graph
        logits_tmp, _ = network_fn(processed_images)

        # create embeddings
        prelogits = slim.fully_connected(logits_tmp, self._embeddings_size, activation_fn=None, scope='Bottleneck', reuse=False)

        logits = slim.fully_connected(prelogits, self.num_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(0.00001),
                                      scope='Logits', reuse=False)

        self._embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # add softmax node add by cjt@20180224
        self._probabilities = tf.nn.softmax(logits, name='predicts')

    # define network architectures
    networks_map = {
        'inception_resnet_v1': create_architecture_inceptionv1,
        'resnet_v1_50': create_architecture_resnet_v1_50,
        'resnet_v2_50': create_architecture_resnet_v2_50,
        'inception_v2': create_architecture_resnet_v2_50,
    }

    # ini method
    def __init__(self, net_name, model_path, num_classes=1000):

        self.model_path       = model_path
        self.num_classes      = num_classes
        self._predictions     = {}
        self._image_size      = default_image_size
        self._embeddings_size = default_embedding_size
        
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

        self.net_name = net_name
        # load network
        if net_name not in self.networks_map:
            raise NotImplementedError

        with self.sess.as_default():

            # define user network
            self.networks_map[self.net_name](self)

            # load weights
            saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
            saver.restore(self.sess, self.model_path)

            # for op in tf.get_default_graph().get_operations():
            #     print(op.name)

        saver = tf.train.Saver()
        saver.save(self.sess, '../tmp/model.ckpt-50')

    # define object classify method
    def object_classify(self, image):

        # null
        if image is None:
            print("object_classify function input image not found")
            return []

        # predict object prob
        self._predictions["cls_prob"] = self.sess.run(self._probabilities, feed_dict={self._image: image})

        # sort prob and index
        top_5_inds,top_5_prob = process_top5_inds_prob(self._predictions["cls_prob"])

        return top_5_inds[0], top_5_prob[0]

# test code
if __name__ == '__main__':

    print(tf.__version__)

    # input parameters
    # check point
    #checkpoints_dir   = '../models/brands/20180227-142155/model-20180227-142155.ckpt-10000'
    checkpoints_dir = '../models/brands/20180228-104446/model-20180228-104446.ckpt-10000'
    # class_num
    class_num_support = 4

    # input one image for test
    image_path        = '../data/5种车款训练样本集/MG-MG3-A款/1015_10045_苏DGR981_0000_1385-1363-0087-0018_37_0_0_0.jpg'

    # create object-classifier instance
    brand_classifier = ObjectClassifier('resnet_v1_50', checkpoints_dir, class_num_support)

    # read one image
    im = cv2.imread(image_path)
    # BGR->RGB
    im = np.array(im)[:, :, ::-1]

    start_time = time.time()
    # classify object
    cls, prob = brand_classifier.object_classify(im)
    end_time = time.time()

    # print result
    print("net cls = %d, prob = %f" % (cls, prob))
    print("net cost time: %f"%(end_time-start_time))








