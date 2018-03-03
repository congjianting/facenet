# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf8")
"""

函数功能

通过多个CKPT模型的预测结果进行对比分析,将错误图片文件筛选出来,然后通过人工进行复核确认真值.

"""

import os
import shutil

# 输入参数
# result.txt预测结果文件的路径
input_results_list = []

# 定义多个预测结果路径
results_path1      = u"/Users/congjt/facenet/data/4种车款训练样本集/result.txt"
results_path2      = u"/Users/congjt/facenet/data/4种车款训练样本集/result2.txt"
results_path3      = u"/Users/congjt/facenet/data/4种车款训练样本集/result3.txt"

# 定义输入图片的根路径
input_image_rootdir   = u"/Users/congjt/facenet/data/4种车款训练样本集"

# 输出参数
output_result_rootdir = u'/Users/congjt/XXXX'

# 定义预测结果的词典
predicts_list = []
predicts_ele  = {}

# 创建导出文件夹的路径
def create_export_folder(export_folder):
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

# 定义主函数
def main():

    # 组织输入图像文件路径
    if results_path1 != "":
        input_results_list.append(results_path1)
    if results_path2 != "":
        input_results_list.append(results_path2)
    if results_path3 != "":
        input_results_list.append(results_path3)

    # 创建输出文件夹路径
    create_export_folder(output_result_rootdir)

    # 遍历读取每个模型的预测结果, 保存到词典里, 理论上这些预测结果的记录个数是相同的
    for one_result in input_results_list:

        predicts_ele = {}

        # 读取输入的result.txt
        with open(one_result, 'r') as f:

            head = f.readline().strip('\n')  # 读取图片文件的文件头

            for one_line in f.readlines():

                # DS-DS 4-A款/1032_10005_蓝豫A1TX20_0000_0262-1786-0106-0025_0_10_0_0_0.jpg 0 999
                one_line     = one_line.strip('\n')

                # 取出.jpg的名字
                filename = one_line.split('.jpg')[0]+".jpg"

                one_line_arr = one_line.split('.jpg')[1].split(' ')

                predicts_ele[filename] = int(one_line_arr[1]) # 当前图片的标签值

        predicts_list.append(predicts_ele)

    # 将预测结果的错误挑选出来
    ele1 = predicts_list[0]
    ele2 = predicts_list[1]
    ele3 = predicts_list[2]

    for key, value in ele1.items():

        # 目前认定文件的路径的文件夹名称为真值
        # 提取真值
        true_label = int(key.split("/")[0])
        # ele1的预测值
        ele1_value = int(ele1[key])

        # ele2的预测值
        ele2_value = int(ele2[key])
        # ele3的预测值
        ele3_value = int(ele3[key])

        # 判断三个模型的预测结果是否都正确, 如果都正确,则表示无争议
        if true_label == ele1_value and true_label == ele2_value and true_label == ele3_value:

            # 无争议
            continue

        else:

            # 创建2个子文件夹, 一个存放三个模型均有差异的数据, 一个存放两个模型认知一致的数据
            create_export_folder(os.path.join(output_result_rootdir, "diff_all")) # 这里存放按照原标签存储
            create_export_folder(os.path.join(output_result_rootdir, "diff_two")) # 这里存放按照two的标签存储

            if ele1_value != ele2_value and ele1_value != ele3_value and ele2_value != ele3_value:

                # 按照true_label建立
                create_export_folder(os.path.join(output_result_rootdir, "diff_all", str(true_label)))

                # 图片剪切到该路径下
                shutil.move(os.path.join(input_image_rootdir, key),
                            os.path.join(output_result_rootdir, "diff_all", str(true_label), os.path.basename(key)))

            else:

                # 其中2个匹配结果
                vote = ""
                if ele1_value == ele2_value or ele1_value == ele3_value:
                    vote = ele1_value
                elif ele2_value == ele3_value:
                    vote = ele2_value

                # 按照two的投票结果建立
                create_export_folder(os.path.join(output_result_rootdir, "diff_two", str(vote)))

                # 图片剪切到该路径下
                shutil.move(os.path.join(input_image_rootdir, key),
                            os.path.join(output_result_rootdir, "diff_two", str(vote), os.path.basename(key)))

            print("move src filename: %s" % os.path.basename(key))

    return

if __name__ == '__main__':

  main()
  print("prepare to check wrong image files done!")
