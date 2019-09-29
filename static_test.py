# coding: utf-8
# 使用静态图片测试模型性能
# by z0gSh1u @ https://github.com/z0gSh1u

import cv2
from keras.models import load_model
import keras
import numpy as np

model = load_model('trained_model/SIGNNET.h5')

files = ['test_picture/0.jpg',
         'test_picture/3.jpg',
         'test_picture/4.jpg',
         'test_picture/5.jpg',
         'test_picture/ed0.jpg',
         'test_picture/ed2.jpg',
         'test_picture/ed4.jpg']

if __name__ == '__main__':
  for file_path in files:
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(image, (64, 64))
    img = (img.reshape(1, 64, 64, 3)).astype('int32') / 255
    pred = model.predict_classes(img)
    print('For filename: ', file_path)
    print('Predict result: ', pred)
    cv2.imshow("Image Preview", image)
    cv2.waitKey(0)

def predict_an_image(src):
  image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
  img = cv2.resize(image, (64, 64))
  img = (img.reshape(1, 64, 64, 3)).astype('int32') / 255
  pred = model.predict_classes(img)
  pred_detail = model.predict(img)
  pred_detail = np.max(pred_detail)
  return pred, pred_detail