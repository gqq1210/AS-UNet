# AS-UNet-tensorflow

## Overview

### Data
1、Glas<br\>
Glas是MICCAI2015腺体分割挑战赛的公开数据集，由165幅图像组成，这些图像来自16个苏木精和伊红（H&E）染色载玻片的大肠癌组织切片。<br\>
2、DRIVE<br\>
DRIVE发布于2003年，是一个用于血管分割的数字视网膜图像数据集，它由40张图片组成，其中7张显示出轻度早期糖尿病视网膜病变迹象。<br\>
3、MoNuSeg<br\>
MoNuSeg发布于2017年，是通过仔细注释几名患有不同器官肿瘤并且在多家医院被诊断患病的患者的组织图像获得的数据集。它由放大40倍捕获到的H＆E染色的组织图像创建而成，训练集包含30张图像和约22000个核边缘注释的训练数据，测试集包含14张图像和约7000个核边缘注释的测试图像。<br\>
数据集链接: https://pan.baidu.com/s/1708AppEJq8ffzp0maG-3Zg  密码: 21mj<br\>

### Model
![](https://github.com/gqq1210/AS-UNet/blob/b525bd9db20e9a15cdd369521879901a5458c3ff/screenshots/AS-UNet.png)

## How to use
Net    | File
------ | ------
UNet   | unet.py  
UNet++ | unet++.py  
KiU-Net| kiu_model.py
DRU-Net| DRU_net.py  
AS-UNet|subtract_refine_newup.py


### train/test
``python test.py``

### 评价指标
``python ts.py``

### 计算参数
``python parameter.py``










