# coding: utf-8
# Gesture识别数据集加载器
# by z0gSh1u @ https://github.com/z0gSh1u

import numpy as np
import h5py

def load_dataset():
  trian_ds = h5py.File('dataset/train_signs.h5', 'r')
  train_x = np.array(trian_ds["train_set_x"][:])
  train_y = np.array(trian_ds["train_set_y"][:])

  test_ds = h5py.File('dataset/test_signs.h5', 'r')
  test_x = np.array(test_ds["test_set_x"][:])
  test_y = np.array(test_ds["test_set_y"][:])

  classes = np.array(test_ds["list_classes"][:])

  return train_x, train_y, test_x, test_y, classes
