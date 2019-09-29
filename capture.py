# coding: utf-8
# 从摄像头获取图片并识别
# by z0gSh1u @ https://github.com/z0gSh1u

import cv2
from static_test import predict_an_image

cap = cv2.VideoCapture(0)
while True:
  ret, data = cap.read()
  # Cutting the captured image towards a square
  img = data[20:, 200:671]
  cv2.imshow("Capture", img)
  key = cv2.waitKey(500) # capture per 500ms
  if key == 27: # esc
    cv2.destroyAllWindows()
    break
  elif key == ord('p'):
    # press p to predict, maybe a little slow at the first press
    pred, prob = predict_an_image(img)
    print('Predict = {}, Prob = {}'.format(pred, prob))

cap.release()
cv2.waitKey(0)