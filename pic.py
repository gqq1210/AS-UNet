import os
import cv2
import PIL.Image
import numpy as np

# img_path = "/home/yl/PycharmProjects/gqq/kiu-net/Data/test/Image/"
# mask_path = "/home/yl/PycharmProjects/gqq/kiu-net/Data/test/Mask-ok/"
img_path = "/home/yl/PycharmProjects/gqq/kiu-net/DRIVE/test/Image/"
mask_path = "/home/yl/PycharmProjects/gqq/kiu-net/DRIVE/test/Mask-ok/"
edge_path = "/home/yl/PycharmProjects/gqq/kiu-net/DRIVE/train/subtract/"

# p = PIL.Image.open(mask_path + "1.bmp")
# p = np.asarray(p)
# print(p)

img = os.listdir(img_path)
for i in img:
    if(i[0] != '.'):
        print(i)
        p = PIL.Image.open(img_path + i)
        p = np.asarray(p)
        p = cv2.resize(p, (128, 128))
        index = [2, 1, 0]
        p = p[:, :, index]
        i = i.split(".")
        cv2.imwrite("/home/yl/PycharmProjects/gqq/kiu-net/DRIVE/test/image_128/" + i[0] + ".tif", p)





# import tensorflow as tf
# import cv2

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# grid = tf.meshgrid(
#         tf.range(8), tf.range(8), indexing='ij'   #生成网格
#     )
# print(sess.run(grid))
# print('----------')
# grid = tf.stack(grid, axis=-1)   #连接
# grid = tf.cast(grid, 'float32')     #(?,?,2)
# grid = tf.reshape(grid, (-1, 2))
# a = tf.expand_dims(grid, 0)
# a = tf.tile(a, [4, 1, 1])
# print(a)

#
#
# x = cv2.imread("/Users/baoqizhao/PycharmProjects/test/gqq/kiu-net/image_src.bmp",  cv2.IMREAD_COLOR)
# # x = "/Users/baoqizhao/PycharmProjects/test/gqq/kiu-net/image_src.bmp"
# x = tf.image.decode_jpeg(x)
# _, row, col, channel = x.get_shape().as_list()
# # 函数只支持GPU操作,indices的值是将整个数组flat后的索引，并保持与池化结果一致的shape
# pool, pool_indices = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1],
#                                                   strides=[1, 2, 2, 1], padding='SAME')
#
# row_col_channel = row * col * channel
# row_col = row * col
# # print(xindex.shape)
# xindex = tf.to_float(pool_indices)
# xindex = xindex - (xindex // row_col_channel) * row_col_channel   #转换到一个batch上
# xindex = xindex - (xindex // row_col) * row_col   #转换到一个通道上
# a_x = xindex // row
# a_y = xindex - (xindex//row) * row
# a_x = (a_x % 2) * 2 - 1
# a_y = (a_y % 2) * 2 - 1
# # a_x = tf.to_float(a_x)
# # a_y = tf.to_float(a_y)
# # print(a_x.shape)
# print(sess.run(a_y))



#
# import time
# Y_M_D_H_M_S = '%Y-%m-%d %H:%M:%S'
# Y_M_D = '%Y-%m-%d'
# H_M_S = '%H:%M:%S'
#
# # 获取系统当前时间并转换请求数据所需要的格式
# def get_time(format_str):
#     # 获取当前时间的时间戳
#     now = int(time.time())
#     # 转化为其他日期格式， 如 “%Y-%m-%d %H:%M:%S”
#     timeStruct = time.localtime(now)
#     strTime = time.strftime(format_str, timeStruct)
#     return strTime
#
# # 获取时分秒时间格式字符串
# def get_hms_time():
#     return get_time(H_M_S)
# print("获取时分秒时间格式字符串: %s" % get_hms_time())

# import os
# import cv2
#
# path_mask_ok = "/home/yl/PycharmProjects/gqq/kiu-net/Data/test/Mask-ok/"
# path_mask_unet = "/home/yl/PycharmProjects/gqq/kiu-net/Data/test/Mask/"
# path_save = "/home/yl/PycharmProjects/gqq/kiu-net/Data/test/subtract/"
# img = "/home/yl/PycharmProjects/gqq/kiu-net/Data/train/Mask/0.bmp"
# a = cv2.imread(img)
# print(a.shape)

# for i in range(130, 160):
#     img_ok = cv2.imread(path_mask_ok + str(i) + ".bmp")
#     img_net = cv2.imread(path_mask_unet + str(i) + ".bmp")
#     subtract = img_net - img_ok
#     cv2.imwrite(path_save + str(i) + ".bmp", subtract)



import cv2
import numpy as np
import os
import PIL.Image
# path = "/home/yl/PycharmProjects/gqq/kiu-net/Data/train/Mask/"
# subtract_path = "/home/yl/PycharmProjects/gqq/kiu-net/Data/test/subtract/"
# path = "/home/yl/PycharmProjects/gqq/kiu-net/DRIVE/train/Mask/"
# subtract_path = "/home/yl/PycharmProjects/gqq/kiu-net/DRIVE/train/subtract/"
# img_path = path + "135.bmp"
# a = cv2.imread(img_path)
# print(a)
# w = 584
# h = 565
#
#
# list = os.listdir(path)
# for kk in list:
#     if kk[0] != ".":
#         img = PIL.Image.open(path + kk)
#         img = np.asarray(img)
#         print(img.shape)
#         subtract = np.zeros(shape=(w, h))
#         #
        # for i in range(512):
        #     for j in range(512):
        #         subtract[i][j][0] = 255
        #         if i == 0 or j == 0 or i == 511 or j == 511:
        #             if img[i][j][0] == 255:
        #                 subtract[i][j][0] = 0
        #         elif img[i][j][0] == 255:
        #             flag = 0
        #             if (img[i - 1][j][0] == 255 or img[i + 1][j][0] == 255 or img[i][j - 1][0] == 255 or img[i][j + 1][
        #                 0] == 255 or img[i - 1][j - 1][0] == 255 or img[i - 1][j + 1][0] == 255 or img[i + 1][j + 1][
        #                 0] == 255 or img[i + 1][j - 1][0] == 255):
        #                 flag = flag + 1
        #             if (img[i - 1][j][0] == 0 or img[i + 1][j][0] == 0 or img[i][j - 1][0] == 0 or img[i][j + 1][
        #                 0] == 0 or img[i - 1][j - 1][0] == 0 or img[i - 1][j + 1][0] == 0 or img[i + 1][j + 1][
        #                 0] == 0 or img[i + 1][j - 1][0] == 0):
        #                 flag = flag + 1
        #             if flag == 2:
        #                 subtract[i][j][0] = 0
        #             else:
        #                 subtract[i][j][0] = 255
        # for i in range(w):
        #     for j in range(h):
        #         subtract[i][j] = 255
        #         if i==0 or j==0 or i==w-1 or j==h-1:
        #             if img[i][j] == 255:
        #                 subtract[i][j] = 0
        #         elif img[i][j] == 255:
        #             flag = 0
        #             if(img[i-1][j] == 255 or img[i+1][j] == 255 or img[i][j-1] == 255 or img[i][j+1] == 255 or img[i-1][j-1] == 255 or img[i-1][j+1] == 255 or img[i+1][j+1] == 255 or img[i+1][j-1] == 255):
        #                 flag = flag + 1
        #             if (img[i - 1][j] == 0 or img[i + 1][j] == 0 or img[i][j - 1] == 0 or img[i][j + 1]
        #                  == 0 or img[i - 1][j - 1] == 0 or img[i - 1][j + 1] == 0 or img[i + 1][j + 1]
        #                  == 0 or img[i + 1][j - 1] == 0):
        #                 flag = flag + 1
        #             if flag == 2:
        #                 subtract[i][j] = 0
        #             else:
        #                 subtract[i][j] = 255
        # # # print(subtract)
        # print(kk)
        # kk = kk.split('.')
        # cv2.imwrite(subtract_path + kk[0] + ".bmp", subtract)






# path = "/home/yl/PycharmProjects/gqq/kiu-net/Data/test/Mask/131.bmp"
# a = PIL.Image.open(path)
# a = np.array(a).copy().astype(np.float32)
# print(a.shape)
# a.reshape(512, 512, 1)
# print(a.shape)

#
# import random
# import shutil

# img_path = "/home/yl/Downloads/Segment/Add_Image/"
# mask_path = "/home/yl/Downloads/Segment/add_Mask/"
#
# img_dest = "/home/yl/PycharmProjects/gqq/kiu-net/Segment/train/Image/"
# mask_dest = "/home/yl/PycharmProjects/gqq/kiu-net/Segment/train/Mask/"
# imgs = os.listdir(img_dest)
# i = 0
# filenumber=len(imgs)
# rate=0.2    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
# picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
# sample = random.sample(imgs, picknumber)  #随机选取picknumber数量的样本图片
# print (sample)
# for img in sample:
#     if(img[0] != '.'):
#         print(img)
#         shutil.move(img_path + img, img_dest + img)
#         shutil.move(mask_path + img, mask_dest + img)

# i = 61
# for img in imgs:
#     if(img[0] != '.'):
#         os.rename(img_dest + img, img_dest + str(i) + ".bmp")
#         os.rename(mask_dest + img, mask_dest + str(i) + ".bmp")
#         i = i + 1



