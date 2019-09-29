# 手势数字识别

利用浅的卷积神经网络（CNN）进行人的手势数字（0~5）识别。使用的框架是后端为Tensorflow的Keras。

### 数据集

`train_signs.h5`与`test_signs.h5`。提供方为`Andrew Ng.`。出于版权原因上传，请自行寻找。

### 标准手势

![1569766118328](https://s2.ax1x.com/2019/09/29/uGDcxP.jpg)

### 网络结构

<img src="https://s2.ax1x.com/2019/09/29/uGDoPs.png" style="zoom:50%;" />

### 超参数

```python
EPOCH = 200
BATCH_SIZE = 64
LR = 0.001
OPTIMIZER = Adam(lr=LR)
LOSSFUNC = 'categorical_crossentropy'
VALIDATION_SPLIT = 0.1
```

### 训练结果

<img src="https://s2.ax1x.com/2019/09/29/uGD72q.png" style="zoom:50%;" />

<img src="https://s2.ax1x.com/2019/09/29/uGDTGn.png" style="zoom:50%;" />

### How-to-use

| 目录/文件          | 说明                                                         |
| ------------------ | ------------------------------------------------------------ |
| /test_picture/     | 提供用于测试的手势图片<br>尽量为正方形，代码中会resize到64*64 |
| /trained_model/    | 训练好的模型，似乎存在泛化能力较差的问题                     |
| dataset_loader.py  | 数据集加载器                                                 |
| dataset_preview.py | 预览数据集                                                   |
| SignNet.py         | 网络构建                                                     |
| static_test.py     | 静态图片测试脚本                                             |
| capture.py         | 从摄像头获取图片并识别<br>白色背景与自然光照会改善识别效果   |

