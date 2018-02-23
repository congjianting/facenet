#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf8")
import os
import time

# 定义耗时函数
def _time_cost(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        fn(*args, **kwargs)
        print "%s cost %s second"%(fn.__name__, time.clock() - start)
    return _wrapper

# 创建文件夹路径
def _create_dst_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# 遍历解析获得文件夹下的json列表
def _each_file_path(rootdir, ext):

    list = os.listdir(rootdir)
    path = []

    for i in range(0, len(list)):
        if os.path.splitext(list[i])[1] == ext:  # '.meta'
            path.append(os.path.join(rootdir, list[i]))
    return path

# 遍历解析获得文件夹下的json列表
def _each_file_name(rootdir, ext):

    list = os.listdir(rootdir)
    path = []

    for i in range(0, len(list)):
        if os.path.splitext(list[i])[1] == ext:  # '.meta'
            path.append(list[i])
    return path

# 递归遍历深层目录下的指定扩展名的文件路径列表
def _dir_0_180_list(path, allfile, ext):
    filelist = os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            # 对于45、135与90度的图像数据文件不进一步遍历
            if filename==u'45' or filename==u'135' or filename==u'90':
                continue
            else:
                _dir_0_180_list(filepath, allfile, ext)
        else:
            if os.path.splitext(filename)[1] == ext:  # '.meta'
                allfile.append(filepath)
    return allfile


# 递归遍历深层目录下的指定扩展名的文件路径列表
def _dir_list(path, allfile, ext):
    filelist = os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            _dir_list(filepath, allfile, ext)
        else:
            if os.path.splitext(filename)[1] == ext:  # '.meta'
                allfile.append(filepath)
    return allfile

# 定义LIST去重的方法
def _del_repeat(liebiao):
    for x in liebiao:
        while liebiao.count(x) > 1:
            del liebiao[liebiao.index(x)]
    return liebiao

