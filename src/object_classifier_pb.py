#-*- coding: UTF-8 -*-
"""See pb predict function
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import cv2
import time
import os
import numpy as np

reload(sys)
sys.setdefaultencoding("utf8")

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

    return top_5_inds, top_5_prob

# define input node name mapping
input_nodename_map = {
                      'resnet_v1_50':        'Placeholder:0',
                      'inception_resnet_v1': 'Placeholder:0',
                     }

# define fea node name mapping
fea_nodename_map = {
                    'resnet_v1_50':        'resnet_v1_50/pool5:0',
                    'inception_resnet_v1': 'embeddings:0',
                   }

# define logits node name mapping
logit_nodename_map = {
                      'resnet_v1_50':        'Softmax:0',
                      'inception_resnet_v1': 'predicts:0',
                     }

class ObjectClassifier_pb:

    # ini method
    def __init__(self, net_name, pb_model_path, phase_train_switch=False):

        self.model_path          = pb_model_path
        self.net_name            = net_name
        self.phase_train_switch  = phase_train_switch
        self._predictions        = {}

        # load network
        with tf.gfile.FastGFile(self.model_path, 'rb') as f:
            output_graph_def = tf.GraphDef()
            output_graph_def.ParseFromString(f.read())
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(output_graph_def, name="")
                self._input_images = graph.get_tensor_by_name(input_nodename_map[self.net_name])  # add input node
                self._output_fea   = graph.get_tensor_by_name(fea_nodename_map[self.net_name])    # add fea node
                self._output_cls   = graph.get_tensor_by_name(logit_nodename_map[self.net_name])  # add softmax node
                if self.phase_train_switch:
                    self._phase_train = graph.get_tensor_by_name("phase_train:0")
                self._graph           = graph

            # set config
            tfconfig = tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.allow_growth = True
            tfconfig=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
            self.sess = tf.Session(graph=self._graph, config=tfconfig)

    # define object classify method
    def object_classify(self, image):

        # null
        if image is None:
            print("object_classify function input image not found")
            return []

        # read image
        h,w,c    = image.shape
        image_4d = np.reshape(image, [1, h, w, c])

        # make feedict
        feed_dict = {self._input_images: image_4d}
        if self.phase_train_switch:
            feed_dict = {self._input_images: image_4d, self._phase_train: False}

        # predict object prob
        self._predictions["cls_prob"], self._predictions["cls_fea"] = self.sess.run([self._output_cls, self._output_fea],
                                                                                    feed_dict=feed_dict)

        # sort prob and index
        top_5_inds, top_5_prob = process_top5_inds_prob(self._predictions["cls_prob"])

        return top_5_inds[0], top_5_prob[0], self._predictions["cls_prob"], self._predictions["cls_fea"] # edit by cjt@180117

# support multi labels
class ObjectClassifier_multi_pb:

    resnet_v1_50_multi_softmax_name_list = ["Softmax_%d:0" % x for x in range(0, 50)]

    networks_map = {
                    'resnet_v1_50_multi': resnet_v1_50_multi_softmax_name_list,
                   }

    # ini method
    def __init__(self, net_name, pb_model_path, num_classes_list):
        self.model_name        = net_name
        self.model_path        = pb_model_path
        self.num_classes_list  = num_classes_list
        self.num_labels        = len(num_classes_list)
        self._predictions_list = []
        self._output_cls_list  = []

        # check net_name
        if net_name not in self.networks_map:
            raise NotImplementedError

        # # set config
        # tfconfig = tf.ConfigProto(allow_soft_placement=True)
        # tfconfig.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=tfconfig)

        # load network
        with tf.gfile.FastGFile(self.model_path, 'rb') as f:
            output_graph_def = tf.GraphDef()
            output_graph_def.ParseFromString(f.read())
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(output_graph_def, name="")
                self._input_images = graph.get_tensor_by_name('Placeholder:0')
                for idx in range(0,self.num_labels):
                    self._output_cls = graph.get_tensor_by_name(self.networks_map[self.model_name][idx])
                    self._output_cls_list.append(self._output_cls)
                self._graph = graph

            # set config
            tfconfig = tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.allow_growth = True
            tfconfig = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            self.sess = tf.Session(graph=self._graph, config=tfconfig)

    # define object classify method
    def object_classify(self, image):

        top_1_inds_list = []
        top_1_prob_list = []

        # null
        if image is None:
            print("object_classify function input image not found")
            return []

        # predict object prob list
        self._predictions_list = self.sess.run(self._output_cls_list, feed_dict={self._input_images: image})

        for one_predict in self._predictions_list:
            # sort prob and index
            top_5_inds, top_5_prob = process_top5_inds_prob(one_predict)
            top_1_inds_list.append(top_5_inds[0])
            top_1_prob_list.append(top_5_prob[0])

        return top_1_inds_list, top_1_prob_list

# test code
if __name__ == '__main__':
    pb_file_path = '../pb/car_cls241_embeddings_50.pb'
    image_path = '../data/5种车款训练样本集/别克-凯越-A款/head_1024_13768_苏DQT862_0000_1023-1499-0096-0024_37_0_0_1.jpg'
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            # fix nodes
            for node in output_graph_def.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in xrange(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']

            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            # read one image
            im = cv2.imread(image_path)
            # BGR->RGB
            im = np.array(im)[:, :, ::-1]

            # resize to 224*224
            im_re = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)

            start_time = time.time()
            # prediction_result, embeddings = sess.run(["predicts:0", "embeddings:0"],
            #                                          feed_dict={"Placeholder:0":im_re, "Placeholder_1:0":False})

            prediction_result, embeddings = sess.run(["predicts:0", "embeddings:0"],
                                                     feed_dict={"Placeholder:0":im_re})

            #print (prediction_result)
            #print (embeddings)

            # sort prob and index
            top_5_inds, top_5_prob = process_top5_inds_prob(prediction_result)

            print (top_5_inds, top_5_prob)




  


