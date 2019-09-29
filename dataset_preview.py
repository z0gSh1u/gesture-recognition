# coding: utf-8
# 数据集预览，测试loader能否正常工作
# by z0gSh1u @ https://github.com/z0gSh1u

import cv2
from dataset_loader import load_dataset

train_x, train_y, test_x, test_y, classes = load_dataset()

print('Size of train set={}'.format(train_x.shape)) # [number, x, y, channel]
print('Size of test set={}'.format(test_x.shape))
print('Snapping train_y[10]={}'.format(train_y[18]))
cv2.imshow('snap.jpg', train_x[18, :, :, :])
cv2.waitKey()
