import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def MAE(res_path, gt_path):
    # res_path='D:/2020_Work/20200306_Trash_class/Semantic/res/ResNet50_100/'
    # gt_path='D:/2020_Work/20200306_Trash_class/Semantic/data/test/labels/'
    res_list = os.listdir(res_path)
    mae = []
    for i in range(len(res_list)):
        r_name = res_path + res_list[i]
        ### '.png'
        g_name = gt_path + res_list[i][:-4] + '.bmp'
        res = cv2.imread(r_name)
        h, w, _ = res.shape
        res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        res = res / 255

        gt = cv2.imread(g_name)
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        gt = gt / 255
        mae.append(sum(sum(abs(res - gt))) / (h * w))
        print(sum(sum(abs(res - gt))) / (h * w))
        # print(sum(mae) / len(mae))

    return sum(mae) / len(mae)


# print('MAE=',MAE(res_path, gt_path)) ##0.0080

def F(res_path, gt_path):
    # res_path = 'D:/2020_Work/20200306_Trash_class/Semantic/res/ResNet50_100/'
    # gt_path = 'D:/2020_Work/20200306_Trash_class/Semantic/data/test/labels/'
    res_list = os.listdir(res_path)
    P = [0 for _ in range(len(res_list))]
    R = [0 for _ in range(len(res_list))]
    PA = [0 for _ in range(len(res_list))]


    for i in range(len(res_list)):
        r_name = res_path + res_list[i]
        g_name = gt_path + res_list[i][:-4] + '.bmp'
        res = cv2.imread(r_name)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        h, w = res.shape

        gt = cv2.imread(g_name)
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        _, gt = cv2.threshold(gt, 125, 255, cv2.THRESH_BINARY)
        print(i, r_name)
        th = 2 * sum(sum(res)) / (h * w)  ##阈值
        _, res_tmp = cv2.threshold(res, th, 255, cv2.THRESH_BINARY)
        tmp = res_tmp - gt
        FP = sum(sum(tmp == 255))
        FN = sum(sum(tmp == -255))
        TP = sum(sum((res_tmp == gt) & (gt == 255)))

        TN = sum(sum((res_tmp == gt) & (gt == -255)))
        # acc = (TP + TN) / (TP + TN + FP + TN)
        PA[i] = (TP + TN) / (TP + TN + FP + TN)


        P[i] = TP / (TP + FP)
        R[i] = TP / (TP + FN)

    belt2 = 0.3
    p = sum(P) / len(P)
    r = sum(R) / len(R)
    pa = sum(PA) / len(PA)
    Fmeasure = ((1 + belt2) * p * r) / (belt2 * p + r)
    # print(Fmeasure)
    return p, r, Fmeasure
    # return pa

# print(F(res_path, gt_path)) ##p=0.863,r=1,F=0.893

def PR(res_path, gt_path):  ##计算时间有点长，因为每张图片涉及到256个阈值
    # res_path = 'D:/2020_Work/20200306_Trash_class/Semantic/res/ResNet50_100/'
    # gt_path = 'D:/2020_Work/20200306_Trash_class/Semantic/data/test/labels/'
    res_list = os.listdir(res_path)
    P = [[0 for _ in range(len(res_list))] for _ in range(256)]
    R = [[0 for _ in range(len(res_list))] for _ in range(256)]
    e = 1e-3
    for i in range(len(res_list)):
        r_name = res_path + res_list[i]
        # g_name = gt_path + res_list[i][:-4] + '.jpg'
        g_name = gt_path + res_list[i]  # 需注意文件后缀名是否一致
        res = cv2.imread(r_name)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        # print(res.shape)
        h, w = res.shape

        gt = cv2.imread(g_name)
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        _, gt = cv2.threshold(gt, 125, 255, cv2.THRESH_BINARY)
        print(i, r_name)
        for j in range(256):
            _, res_tmp = cv2.threshold(res, j, 255, cv2.THRESH_BINARY)
            tmp = res_tmp - gt
            FP = sum(sum(tmp == 255))
            FN = sum(sum(tmp == -255))
            TP = sum(sum((res_tmp == gt) & (gt == 255)))
            # print(TP,FP,FN)
            P[j][i] = TP / (TP + FP + e)
            R[j][i] = TP / (TP + FN + e)

    F = []
    belt2 = 0.3
    p_tmp = []
    r_tmp = []
    for j in range(256):  ##每个阈值下的p,r,F值
        p = sum(P[j]) / len(P[j])
        r = sum(R[j]) / len(R[j])
        # print(j, p, r)
        Fmeasure = ((1 + belt2) * p * r) / (belt2 * p + r)
        F.append(Fmeasure)
        p_tmp.append(p)
        r_tmp.append(r)
    # npy_save_path = './npy/'
    # if not os.path.exists(npy_save_path):
    #     os.makedirs(npy_save_path)
    # np.save(npy_save_path + 'P.npy', P)
    # np.save(npy_save_path + 'R.npy', R)
    # np.save(npy_save_path + 'p_tmp.npy', p_tmp)
    # np.save(npy_save_path + 'r_tmp.npy', r_tmp)

    # ##绘制PR曲线
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(r_tmp[:-1], p_tmp[:-1])
    plt.savefig("pr.png")
    plt.show()
    return max(F), min(F)


# res_path = 'D:/2020_Work/20200306_Trash_class/Semantic/res/ResNet50_100/'
# gt_path = 'D:/2020_Work/20200306_Trash_class/Semantic/data/test/labels/'
# print(PR(res_path, gt_path))

def dice(res_path, gt_path):
    # res_path = 'D:/2020_Work/20200306_Trash_class/Semantic/res/ResNet50_100/'
    # gt_path = 'D:/2020_Work/20200306_Trash_class/Semantic/data/test/labels/'
    res_list = os.listdir(res_path)
    di = []
    for i in range(len(res_list)):
        r_name = res_path + res_list[i]
        g_name = gt_path + res_list[i][:-4] + '.bmp'
        res = cv2.imread(r_name)
        h, w, _ = res.shape
        res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        _, res = cv2.threshold(res, 125, 255, cv2.THRESH_BINARY)

        gt = cv2.imread(g_name)
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        _, gt = cv2.threshold(gt, 125, 255, cv2.THRESH_BINARY)

        iou_and = sum(sum((res == 255) & (gt == 255)))
        iou_or = sum(sum(res == 255)) + sum(sum(gt == 255))
        print(2 * iou_and, iou_or)
        di.append(2 * iou_and / iou_or)

    return sum(di) / len(di)


# print('dice=',dice(res_path ,gt_path)) ##dice= 0.9688

def IOU(res_path, gt_path):
    # res_path = 'D:/2020_Work/20200306_Trash_class/Semantic/res/ResNet50_100/'
    # gt_path = 'D:/2020_Work/20200306_Trash_class/Semantic/data/test/labels/'
    res_list = os.listdir(res_path)
    iou = []
    for i in range(len(res_list)):
        r_name = res_path + res_list[i]
        g_name = gt_path + res_list[i][:-4] + '.bmp'
        res = cv2.imread(r_name)
        h, w, _ = res.shape
        res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        _, res = cv2.threshold(res, 125, 255, cv2.THRESH_BINARY)

        gt = cv2.imread(g_name)
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        _, gt = cv2.threshold(gt, 125, 255, cv2.THRESH_BINARY)

        iou_and = sum(sum((res == 255) & (gt == 255)))
        iou_or = sum(sum(res == 255)) + sum(sum(gt == 255)) - iou_and
        print(iou_and, iou_or)
        iou.append(iou_and / iou_or)

    return sum(iou) / len(iou)


# print('IOU=',IOU(res_path, gt_path)) ##IOU= 0.9402

def ROC(res_path, gt_path):  # 与PR曲线类似
    res_list = os.listdir(res_path)
    TPR = [[0 for _ in range(len(res_list))] for _ in range(256)]
    FPR = [[0 for _ in range(len(res_list))] for _ in range(256)]
    e = 1e-3
    for i in range(len(res_list)):
        r_name = res_path + res_list[i]
        g_name = gt_path + res_list[i]  # 需注意文件后缀名是否一致
        res = cv2.imread(r_name)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

        gt = cv2.imread(g_name)
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        _, gt = cv2.threshold(gt, 125, 255, cv2.THRESH_BINARY)
        print(i, r_name)
        for j in range(256):
            _, res_tmp = cv2.threshold(res, j, 255, cv2.THRESH_BINARY)
            tmp = res_tmp - gt
            FP = sum(sum(tmp == 255))
            FN = sum(sum(tmp == -255))
            TP = sum(sum((res_tmp == gt) & (gt == 255)))
            TN = sum(sum((res_tmp == gt) & (gt == -255)))
            TPR[j][i] = TP / (TP + FN + e)
            FPR[j][i] = FP / (FP + TN + e)

    tpr = []
    fpr = []
    for j in range(256):
        tpr.append(sum(TPR[j]) / len(TPR[j]))
        fpr.append(sum(FPR[j]) / len(FPR[j]))
    # ##绘制ROC曲线
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(tpr[:-1], fpr[:-1])
    plt.savefig("ROC.png")
    plt.show()




def get_contours(img):
    """获取连通域

    :param img: 输入图片
    :return: 最大连通域
    """
    # 灰度化, 二值化, 连通域分析
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def Hausdorff(res_path, gt_path):
    res_list = os.listdir(res_path)
    haus = []
    for i in range(len(res_list)):
        r_name = res_path + res_list[i]
        g_name = gt_path + res_list[i][:-4] + '.bmp'
        print(r_name, g_name)
        res = cv2.imread(r_name)
        h, w, _ = res.shape

        gt = cv2.imread(g_name)

        # 2.获取图片连通域
        cnt_cs1 = get_contours(res)
        cnt_cs2 = get_contours(gt)

        # 3.创建计算距离对象
        hausdorff_sd = cv2.createHausdorffDistanceExtractor()

        # 4.计算轮廓之间的距离
        d1 = hausdorff_sd.computeDistance(cnt_cs1, cnt_cs2)
        haus.append(d1)
    return sum(haus) / len(haus)


def MSE(res_path, gt_path):
    # res_path='D:/2020_Work/20200306_Trash_class/Semantic/res/ResNet50_100/'
    # gt_path='D:/2020_Work/20200306_Trash_class/Semantic/data/test/labels/'
    res_list = os.listdir(res_path)
    mse = []
    for i in range(len(res_list)):
        r_name = res_path + res_list[i]
        ### '.png'
        g_name = gt_path + res_list[i][:-4] + '.bmp'
        res = cv2.imread(r_name)
        h, w, _ = res.shape
        res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        res = res / 255

        gt = cv2.imread(g_name)
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        gt = gt / 255
        mse.append(sum(sum(np.square(res - gt))) / (h * w))
        print(sum(sum(np.square(res - gt))) / (h * w))
        # print(sum(mae) / len(mae))

    return sum(mse) / len(mse)




gt_path = '/home/yl/PycharmProjects/gqq/kiu-net/DRIVE/test/Mask-ok/'
res_path = '/home/yl/PycharmProjects/gqq/kiu-net/DRIVE/test/Mask_FCN/'
F = F(res_path, gt_path)
IOU = IOU(res_path, gt_path)
mse = MSE(res_path, gt_path)
Hausdorff = Hausdorff(res_path, gt_path)
dice = dice(res_path, gt_path)
print("p=", F[0])
print("r=", F[1])
print("IOU=", IOU)
print("MSE=", mse)
print("dice=", dice)
print("Hausdorff=", Hausdorff)  # Hausdorff= 65.96012181915597

