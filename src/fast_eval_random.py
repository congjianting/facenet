#encoding:utf-8
from __future__ import print_function

from scipy import misc
import sys, time
import math, random
import os
import os.path as osp
import yaml
import numpy as np
import datetime
import cv2

from object_classifier import ObjectClassifier

def compareFeatures(feature1, feature2):
    dist = np.sqrt(np.sum(np.square(np.subtract(feature1, feature2))))
    return dist

def eval(classifier, image_files, objects, thredholds):
    nrof_images = len(image_files)
    nrof_thredholds = len(thredholds)

    tn = np.zeros(nrof_thredholds)
    fn = np.zeros(nrof_thredholds)
    tp = np.zeros(nrof_thredholds)
    fp = np.zeros(nrof_thredholds)

    print("Extract Features:")
    features = []
    for i in range(nrof_images):
        print(image_files[i])
        im = cv2.imread(image_files[i])
        # BGR->RGB
        im = np.array(im)[:, :, ::-1]

        # embedding feature extract
        feature = classifier.object_feature(im) # 128 dim

        features.append(feature)

    print('Compare Images:')
    for i in range(0, nrof_images - 1, 10):
        print(image_files[i])
        w = i + 1
        while len(objects) > w and objects[i] == objects[w]:
            w += 1
        
        z = i - 1
        while z > 0 and objects[i] == objects[z]:
            z -= 1

        for j in range(z, w):
            if random.random() < 0.1: # 5% to test 
                if i != j and features[i] is not None and features[j] is not None:
                    distance = compareFeatures(features[i], features[j])
                    issame = True
                    for k in range(nrof_thredholds):
                        positive = distance < thredholds[k]

                        if positive:
                            if issame:
                                tp[k] += 1
                            else:
                                fp[k] += 1
                        else:
                            if issame:
                                fn[k] += 1
                            else:
                                tn[k] +=1

        for w in range(0, 30):
            j = int(random.random()  * nrof_images)
            if objects[i] == objects[j]:
                continue

            if features[i] is not None and features[j] is not None:
                distance = compareFeatures(features[i], features[j])
                issame = False
                for k in range(nrof_thredholds):
                    positive = distance < thredholds[k]

                    if positive:
                        if issame:
                            tp[k] += 1
                        else:
                            fp[k] += 1
                    else:
                        if issame:
                            fn[k] += 1
                        else:
                            tn[k] +=1

    
    same_count = tp[0] + fn[0]
    total_count = tp[0] + tn[0] + fp[0] + fn[0]
    print("Total: {}, same: {}".format(total_count, same_count))
    print('')
    print('                  Accuarcy    Validate     Recall      F-Score      FPR')
    for k in range(nrof_thredholds):
        precision = 1.0*tp[k]/(tp[k]+fp[k])
        TPR=recall = 1.0*tp[k]/same_count
        fscore = 2*precision*recall/(precision + recall)
        FPR = fp[k] / (fp[k] + tn[k])
        print('Thredhold %1.2f:    %1.4f,     %1.4f,    %1.4f,    %1.4f,    %1.4f' % (
            thredholds[k], 
            1.0*(tp[k] + tn[k]) / total_count, 
            precision, recall, fscore, FPR))

def get_images(data_path):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    objects = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for root, dirnames, _ in os.walk(data_path):
        for dirname in dirnames:
            folder_path = os.path.join(root, dirname)
            for parent, _, filenames in os.walk(folder_path):
                for filename in filenames:
                    for ext in exts:
                        if filename.endswith(ext):
                            files.append(os.path.join(parent, filename))
                            objects.append(dirname)
                            break
    print('Find {} images'.format(len(files)))
    return files, objects

if __name__ == '__main__':

    import numpy as np

    print(np.__version__)

    # input images path
    image_root_path   = u"../data/4种车款训练样本集"

    # check point
    checkpoints_dir   = '../models/brands/20180301-114913/model-20180301-114913.ckpt-41284'
    # class_num
    class_num_support = 241

    # create object-classifier instance
    classifier        = ObjectClassifier('inception_resnet_v1', checkpoints_dir, class_num_support, exclude_var='Logits')

    files, objects    = get_images(image_root_path)

    eval(classifier, files, objects, np.arange(0, 2.0, 0.01))

    print("run eval done!")