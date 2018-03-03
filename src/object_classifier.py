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

import sys
reload(sys)
sys.setdefaultencoding("utf8")

from nets          import nets_factory

def _dir_list(path, allfile, ext):

    filelist = os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            _dir_list(filepath, allfile, ext)
        else:
            #
            ext_array = ext.split(u'|')
            for i in ext_array:
                if os.path.splitext(filename)[1] == i.replace(u' ','',5):  # '.meta'
                    allfile.append(filepath)
    return allfile

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

    # define densenet121 network
    def create_architecture_densenet_121(self):
        # define input image placeholder
        self._image = tf.placeholder(tf.uint8, shape=[None, None, 3])

        # resize
        self._imagere = tf.image.resize_images(self._image, (self._image_size, self._image_size))

        # extend dim
        processed_images = tf.expand_dims(tf.image.per_image_standardization(self._imagere), 0)

        network_fn = nets_factory.get_network_fn(
                                                'densenet121',
                                                num_classes=None,
                                                weight_decay=0.00004,
                                                is_training=False)

        # Build the inference graph
        logits_tmp, _ = network_fn(processed_images)

        # create embeddings
        prelogits = slim.fully_connected(logits_tmp, self._embeddings_size, activation_fn=None, scope='Bottleneck',
                                         reuse=False)

        logits = slim.fully_connected(prelogits, self.num_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(0.00001),
                                      scope='Logits', reuse=False)

        self._embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # add softmax node add by cjt@20180224
        self._probabilities = tf.nn.softmax(logits, name='predicts')

    # define densenet161 network
    def create_architecture_densenet_161(self):
        # define input image placeholder
        self._image = tf.placeholder(tf.uint8, shape=[None, None, 3])

        # resize
        self._imagere = tf.image.resize_images(self._image, (self._image_size, self._image_size))

        # extend dim
        processed_images = tf.expand_dims(tf.image.per_image_standardization(self._imagere), 0)

        network_fn = nets_factory.get_network_fn(
                                                'densenet161',
                                                num_classes=None,
                                                weight_decay=0.00004,
                                                is_training=False)

        # Build the inference graph
        logits_tmp, _ = network_fn(processed_images)

        # create embeddings
        prelogits = slim.fully_connected(logits_tmp, self._embeddings_size, activation_fn=None, scope='Bottleneck',
                                         reuse=False)

        logits = slim.fully_connected(prelogits, self.num_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(0.00001),
                                      scope='Logits', reuse=False)

        self._embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # add softmax node add by cjt@20180224
        self._probabilities = tf.nn.softmax(logits, name='predicts')

    # define densenet169 network
    def create_architecture_densenet_169(self):
        # define input image placeholder
        self._image = tf.placeholder(tf.uint8, shape=[None, None, 3])

        # resize
        self._imagere = tf.image.resize_images(self._image, (self._image_size, self._image_size))

        # extend dim
        processed_images = tf.expand_dims(tf.image.per_image_standardization(self._imagere), 0)

        network_fn = nets_factory.get_network_fn(
                                                'densenet169',
                                                num_classes=None,
                                                weight_decay=0.00004,
                                                is_training=False)

        # Build the inference graph
        logits_tmp, _ = network_fn(processed_images)

        # create embeddings
        prelogits = slim.fully_connected(logits_tmp, self._embeddings_size, activation_fn=None, scope='Bottleneck',
                                         reuse=False)

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
                    'inception_v2': create_architecture_inception_v2,
                    'densenet_121': create_architecture_densenet_121,
                    'densenet_161': create_architecture_densenet_161,
                    'densenet_169': create_architecture_densenet_169,
    }

    # ini method
    def __init__(self, net_name, model_path, num_classes=1000, exclude_var=None):

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
            all_vars       = tf.trainable_variables()
            var_to_restore = all_vars
            if exclude_var is not None:
                var_to_restore = [v for v in all_vars if not v.name.startswith(exclude_var)]

            saver = tf.train.Saver(var_to_restore, max_to_keep=3)
            saver.restore(self.sess, self.model_path)

            for op in tf.get_default_graph().get_operations():
                print(op.name)

        saver = tf.train.Saver()
        saver.save(self.sess, '../tmp/model.ckpt-50')

    # define object classify method
    def object_classify(self, image):

        # null
        if image is None:
            print("object_classify function input image not found")
            return []

        # predict object prob
        self._predictions["cls_prob"], self._predictions["cls_fea"] = self.sess.run([self._probabilities, self._embeddings], feed_dict={self._image: image})

        # sort prob and index
        top_5_inds,top_5_prob = process_top5_inds_prob(self._predictions["cls_prob"])

        return top_5_inds[0], top_5_prob[0], self._predictions["cls_fea"]

    # define object feature extraction method
    def object_feature(self, image):

        # null
        if image is None:
            print("object_feature function input image not found")
            return []

        # predict object feature
        self._predictions["cls_fea"] = self.sess.run([self._embeddings], feed_dict={self._image: image})

        return self._predictions["cls_fea"]

# test code
if __name__ == '__main__':

    print(tf.__version__)

    # define batch debug switch
    _debug_batch_switch = True

    # check point
    checkpoints_dir   = '../models/brands/20180228-104446/model-20180228-104446.ckpt-10000'
    # class_num
    class_num_support = 4

    # create object-classifier instance
    classifier = ObjectClassifier('resnet_v1_50', checkpoints_dir, class_num_support)

    # define use true label dict
    use_true_label_dict = True

    # 定义真值标签词典
    true_label = {
                    u"MG-MG3-A款": 1,
                    u"DS-DS 4-A款": 0,
                    u"本田-飞度-A款": 2,
                    u"比亚迪-秦-A款": 3,
    }


    if _debug_batch_switch == False:

        # input parameters
        # input one image for test
        image_path        = '../data/5种车款训练样本集/MG-MG3-A款/1015_10045_苏DGR981_0000_1385-1363-0087-0018_37_0_0_0.jpg'

        # read one image
        im = cv2.imread(image_path)
        # BGR->RGB
        im = np.array(im)[:, :, ::-1]

        start_time = time.time()
        # classify object
        cls, prob, _ = classifier.object_classify(im)
        end_time = time.time()

        # print result
        print("net cls = %d, prob = %f" % (cls, prob))
        print("net cost time: %f"%(end_time-start_time))

    else:

        # define batch images path

        # format
        file_ex = u'.jpeg | .jpg'

        # 图像文件的根路径
        input_image_rootdir = u'../data/test'

        image_path_list = []
        # 遍历预测的图像根路径
        _dir_list(input_image_rootdir, image_path_list, file_ex)

        # 定义统计单个类别召回率的指标和平均识别率指标
        total_right = 0
        total_all   = 0
        trues       = {}
        precisons   = {}

        total_error_confused = 0
        confused_cred        = 0.95

        with open(os.path.join(input_image_rootdir, u"result.txt"), u"w+") as f:

            f.write(input_image_rootdir)  # 写入根路径

            for image_path in image_path_list:

                # read one image
                im = cv2.imread(image_path)
                # BGR->RGB
                im = np.array(im)[:, :, ::-1]

                # 运行输入图像来计算各类别概率
                cls, prob, _ = classifier.object_classify(im)

                # 预测结果写入到result.txt中
                # 提取出文件的相对路径
                relative = image_path[len(input_image_rootdir) + 1:]
                strresult = "\n%s %d %d" % (relative, cls, int(prob * 1000))
                f.write(strresult)  # 写入当前图片预测结果

                # 打印显示
                print(strresult)

                # 统计识别效果
                folder_path   = os.path.dirname(image_path)
                folder_arr    = folder_path.split('/')
                label_chinese = folder_arr[len(folder_arr)-1]

                # 统计真值
                if label_chinese in trues.keys():
                    trues[label_chinese] += 1
                else:
                    trues[label_chinese]  = 1

                # 统计当前图片的命中率
                if use_true_label_dict: # 使用对应词典

                    if label_chinese in precisons.keys():
                        if cls == true_label[label_chinese]:
                            precisons[label_chinese] += 1
                    else:
                        precisons[label_chinese]  = 0
                        if cls == true_label[label_chinese]:
                            precisons[label_chinese] += 1

                else:

                    # 不使用词典, 认为文件夹名称就是真值编号
                    if label_chinese in precisons.keys():
                        if cls == int(label_chinese):
                            precisons[label_chinese] += 1
                    else:
                        precisons[label_chinese]  = 0
                        if cls == int(label_chinese):
                            precisons[label_chinese] += 1

                # 统计平均识别率
                if use_true_label_dict:  # 使用对应词典

                    if cls == true_label[label_chinese]:
                        total_right += 1

                    # 预测结果错误,但是置信度高于阈值
                    if cls != true_label[label_chinese] and prob > confused_cred:
                        total_error_confused += 1

                else:

                    if cls == int(label_chinese):
                        total_right += 1

                    # 预测结果错误,但是置信度高于阈值
                    if cls != int(label_chinese) and prob > confused_cred:
                        total_error_confused += 1

                total_all += 1

        # 预测结束后打印各个指标
        # 打印当前的各个类别的召回率
        for key, value in trues.items():

            # 当前类别的召回率指标
            print("key: %s====>%f, total===>%d, right===>%d\n" % (key, 1.0*precisons[key]/trues[key], trues[key], precisons[key]))

        # 打印平均识别率
        print("average precison: %f, diff: %f total====>%d, right=====>%d, confused=====>%d,\n" % (1.0*total_right/total_all, 1.0*total_error_confused/total_all,
                                                                                                   total_all, total_right, total_error_confused))






