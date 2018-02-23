#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import os
from PIL import Image
import  shutil
from common_fun import _dir_list
import cv2

""" function

check image data is okay

"""

# 输入参数
input_path = u'/opt/congjt/project/download/youxin_image'

# 输出参数
error_path = u'/opt/congjt/project/download/trainorerror_image'

#
image_file_path_list = []
_dir_list(input_path, image_file_path_list, u'.jpg')
right_file_counter = 0
error_file_counter = 0
total_file_counter = 0

for image_file_path in image_file_path_list:

    #
    try:

        im = Image.open( image_file_path ) #

        img1 = cv2.imread( image_file_path, cv2.IMREAD_COLOR )  #
        if img1 is None:
            print image_file_path + u' jpg data error!'
            error_file_counter += 1
            #
            shutil.move(image_file_path, os.path.join(error_path, os.path.basename(image_file_path)))
        else:
            right_file_counter += 1

            # 重新写入文件
            if im.format != u"JPEG":
                print u'文件不是JPEG格式：' + image_file_path
                cv2.imwrite( image_file_path, img1 )

    except:
        print image_file_path + u' open error!'
        error_file_counter += 1
        #
        shutil.move( image_file_path, os.path.join(error_path, os.path.basename(image_file_path)) )
    finally:
        #print image_file_path + u' open!'
        total_file_counter += 1



#
print u'正确读取的文件数目: ' + str(right_file_counter)
print u'错误读取的文件数目: ' + str(error_file_counter)
print u'total读取的文件数目: ' + str(total_file_counter)
print u'文件夹数目: ' + str(len(image_file_path_list))