from __future__ import division
from uNet.FCN1 import unet2dModule
import numpy as np
import pandas as pd
import cv2
import  os
import PIL.Image
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'




def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvmaskdata = pd.read_csv('./DRIVE/GlandsMask.csv')
    csvimagedata = pd.read_csv('./DRIVE/GlandsImage.csv')
    csvimageedge = pd.read_csv('./DRIVE/GlandsMask.csv')   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values
    edge = csvimageedge.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]
    edge = edge[perm]
    unet2d = unet2dModule(512, 512, channels=3, costname="dice coefficient")
    unet2d.train(imagedata, maskdata, "/home/yl/PycharmProjects/gqq/kiu-net/Data/model/FCN/unet2dglandceil.pd",
                 "log/", 0.0005, 0.8, 100000, 2)
    #unet2d.train(imagedata, maskdata, edge, "/home/yl/PycharmProjects/gqq/kiu-net/Data8580/test_loss/unet2dglandceil.pd",
    #             "log/", 0.0005, 0.8, 100000, 2)


def predict():
    for i in range(1, 11):
        # true_img = PIL.Image.open("/home/yl/PycharmProjects/gqq/kiu-net/Data/test/Image/"+ str(i) + ".bmp")
        # true_img = np.asarray(true_img)
        true_img = cv2.imread("/home/yl/PycharmProjects/gqq/kiu-net/DRIVE/test/Image/"+ str(i) + ".tif", cv2.IMREAD_COLOR)
        # subtract_img = cv2.imread("/home/yl/PycharmProjects/gqq/kiu-net/Data/test/subtract/"+ str(i) + ".bmp", cv2.IMREAD_GRAYSCALE)
        test_images = true_img.astype(np.float)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        # subtract_images = subtract_img.astype(np.float)
        # subtract_images = np.multiply(subtract_images, 1.0 / 255.0)
        unet2d = unet2dModule(512, 512, 3)
        predictvalue = unet2d.prediction("/home/yl/PycharmProjects/gqq/kiu-net/Data/model/FCN/unet2dglandceil.pd2000", test_images)
        cv2.imwrite("/home/yl/PycharmProjects/gqq/kiu-net/Data/test/Mask_FCN/" + str(i) + ".bmp", predictvalue)





    # #true_img = cv2.imread("/home/ubuntu/PycharmProjects/VnetFamily/uNet/Data/test/Image/%D.bmp", cv2.IMREAD_COLOR)
    # #for i in range(136, 150):
    # true_img = cv2.imread("/home/yl/PycharmProjects/gqq/kiu-net/Data/test/image_new/27.bmp", cv2.IMREAD_COLOR)
    # #cv2.imwrite(true_img)
    # test_images = true_img.astype(np.float)
    # # convert from [0:255] => [0.0:1.0]
    # test_images = np.multiply(test_images, 1.0 / 255.0)
    # unet2d = unet2dModule(128, 128, 3)
    # predictvalue = unet2d.prediction("/home/yl/PycharmProjects/gqq/kiu-net/model/unet2dglandceil.pd",
    #                                  test_images)
    #
    # cv2.imwrite("/home/yl/PycharmProjects/gqq/kiu-net/Data/test/Mask/127.bmp", predictvalue)


def main(argv):
    if argv == 1:
        train()
    if argv == 2:
        predict()


if __name__ == "__main__":
    main(2)