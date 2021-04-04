# AS-UNet-tensorflow

## Overview

### Data
1、Glas  
	Glas是MICCAI2015腺体分割挑战赛的公开数据集，由165幅图像组成，这些图像来自16个苏木精和伊红（H&E）染色载玻片的大肠癌组织切片。  
2、DRIVE  
	DRIVE发布于2003年，是一个用于血管分割的数字视网膜图像数据集，它由40张图片组成，其中7张显示出轻度早期糖尿病视网膜病变迹象。  
3、MoNuSeg  
	MoNuSeg发布于2017年，是通过仔细注释几名患有不同器官肿瘤并且在多家医院被诊断患病的患者的组织图像获得的数据集。它由放大40倍捕获到的H＆E染色的组织图像创建而成，训练集包含30张图像和约22000个核边缘注释的训练数据，测试集包含14张图像和约7000个核边缘注释的测试图像。  
数据集链接: https://pan.baidu.com/s/1708AppEJq8ffzp0maG-3Zg  密码: 21mj  

### Model
1）提出边缘注意模块，强化边缘，减少边缘缺失值。通过掩膜边缘提取算法得到掩膜边缘图像，连接到UNet扩张路径的最后三层上，以强化边缘信息；并在BAB中引入新的注意力模块，结合通道注意力和空间注意力，激活特征响应，增强图像中关键信息的获取，提升网络对目标区域的分割能力。  
2）提出使用区域和边缘组合损失函数，在提高分割精度的同时，实现测试时参数减少。组合损失函数结合了基于区域的Dice Loss和基于边缘的Boundary Loss，在保证区域缺失值小的同时补充边缘信息，提高分割精度；此外，通过组合损失函数的作用在训练时经过前后向反馈不断更新AS-UNet中的网络参数，使得训练好的模型在测试时可以舍弃添加的BAB部分，降低预测的时间代价。  
![](https://github.com/gqq1210/AS-UNet/blob/b525bd9db20e9a15cdd369521879901a5458c3ff/screenshots/AS-UNet.png)

## How to use
Net    | File
------ | ------
UNet   | unet.py  
UNet++ | unetplus.py  
KiU-Net| kiu_model.py
DRU-Net| DRU_net.py  
AS-UNet|subtract_refine_newup.py


### train/test
``python test.py``

### 评价指标
``python ts.py``

### 计算参数
``python parameter.py``










