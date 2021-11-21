# 这是一个示例 Python 脚本。
# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import numpy as np
import matplotlib.pyplot as plt
from logRegress import *
import glob
import random
import cv2
import tensorflow as tf

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。

def path_to_data(all_img_path):
    train_set = []
    data_label = []
    for path in all_img_path:
        img = cv2.imread(path)
        mat = np.array(cv2.resize(img,(64,64)))
        train_set.append(mat.flatten())
    return train_set


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    train_image_path = glob.glob("dc/train/*.jpg")
    train_cat_path = [s for s in train_image_path if s.split('\\')[-1].split('.')[0]=="cat" ]
    train_dog_path = [s for s in train_image_path if s.split('\\')[-1].split('.')[0]!="cat" ]
    #准备测试集数据
    test_dog_path = train_cat_path[400:550]
    test_cat_path = train_cat_path[500:650]
    test_cat_path.extend(test_dog_path)
    test_label = [s.split('\\')[-1].split('.')[0] for s in test_cat_path]
    test_data = path_to_data(test_cat_path)
    label_to_index = {'dog': 0, 'cat': 1}
    test_labels_nums = [label_to_index.get(l) for l in test_label]
    #####准备训练集数据
    train_cat_path = train_cat_path[:500]
    train_dog_path = train_dog_path[:500]
    train_cat_path.extend(train_dog_path)
    train_image_path = train_cat_path
    random.shuffle(train_image_path)
    train_label = [s.split('\\')[-1].split('.')[0] for s in train_image_path]
    train_labels_nums = [label_to_index.get(l) for l in train_label]
    dataArr = path_to_data(train_image_path)
    errorSum = 0.0
    for k in range(10):
        errorSum += dogvsCatTest(dataArr,train_labels_nums, test_data, test_labels_nums)
    print("%d 次测试，平均错误率为%f" % (10, errorSum / float(10)))



    #dataArr,labelMat = loadDataset()
    # weight = gradAscent(dataArr,labelMat)
    # labelMat = np.array(labelMat)
    # dataArr = np.array(dataArr)
    # data1 = dataArr[labelMat==1]
    # print(data1)
    # print(data1[:,1].tolist())
    # print(weight)
    # weight = stocGradAscent1(np.array(dataArr),labelMat,500)
    # plotBestFit(weight)
    #multiTest(20)
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
