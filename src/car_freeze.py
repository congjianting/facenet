# coding=UTF-8
import tensorflow as tf
import os.path
import argparse
from tensorflow.python.framework import graph_util
import cv2
import numpy as np

MODEL_DIR = "../pb/"
MODEL_NAME = "car_cls241_embeddings_50.pb"

if not tf.gfile.Exists(MODEL_DIR):
    tf.gfile.MakeDirs(MODEL_DIR)

def freeze_graph(model_folder):
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME)

    output_node_names = ["predicts","embeddings"]
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        tensors=graph.get_collection(tf.GraphKeys.VARIABLES)
        # file=open('/Users/youj/result.txt','w')
        # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #     file.write(str(i.name)+"\n")

        image_path = '../data/200种车款训练样本集/DS-DS 4-A款/1032_10007_蓝豫AG7G52_0000_1403-1891-0113-0025_0_3_0_0_0.jpg'
        im = cv2.imread(image_path)
        # BGR->RGB
        im = np.array(im)[:, :, ::-1]

        # resize to 224*224
        im_re = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)

        im_re= np.reshape(im_re, [1,224,224,3])

        print "predictions : ", sess.run(["predicts:0", "embeddings:0"],
                                         feed_dict={"input:0":im_re})

        # print "predictions : ", sess.run(["predicts:0", "embeddings:0"],
        #                                  feed_dict={"Placeholder:0": im_re, "Placeholder_1:0": False})

        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            #output_node_names.split(",")
            output_node_names
        )
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        # for op in graph.get_operations():
        #     print(op.name, op.values())

if __name__ == '__main__':

    freeze_graph("../tmp/")