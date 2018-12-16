import numpy as np
import h5py
import matplotlib.pyplot as plt

def load_dataset():
    train_dataset=h5py.File('datasets/train_catvnoncat.h5',"r")
    train_set_x_orig=np.array(train_dataset["train_set_x"][:])  # 提取训练数据集中的所有特征
    train_set_y_orig=np.array(train_dataset["train_set_y"][:])  # 提取数据集中分类的标签

    test_dataset=h5py.File('datasets/test_catvnoncat.h5',"r")
    test_set_x_orig=np.array(train_dataset["test_set_x"][:])  # 提取测试数据集的特征
    test_set_y_orig=np.array(train_dataset["test_set_y"][:])  # 测试数据集标签

    classes=np.array(test_dataset["list_classes"][:])  # 种类的数量

    train_set_y_orig= np.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig= np.array((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

index=25

train_dataset=h5py.File('datasets/train_catvnoncat.h5',"r")
train_set_x_orig=np.array(train_dataset["train_set_x"][:])  # 提取训练数据集中的所有特征
train_set_y_orig=np.array(train_dataset["train_set_y"][:])  # 提取数据集中分类的标签
plt.show()
