import torch
import cv2
import segmentation_models_pytorch as smp
import numpy as np
import PIL.Image
def get_img(path, image_path):
    # img = cv2.imread(f'{path}/{image_path}')
    img = PIL.Image.open(f'{path}/{image_path}')
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

if __name__ == '__main__':
    for i in range(1, 11):
        path = 'DRIVE/test/Mask_FCN/'
        p1 = get_img(path, f'{i}.bmp')
        p1 = p1.astype(np.float32) / 255
        p1 = torch.from_numpy(p1)

        path = 'DRIVE/test/Mask-ok'
        p2 = get_img(path, f'{i}.bmp')
        p2 = p2.astype(np.float32) / 255
        p2 = torch.from_numpy(p2)

        cri = smp.utils.losses.DiceLoss()
        print(1 - cri(p1, p2).item())

    # for i in range(130, 160):
    #     path = 'Data/test/Mask'
    #     p1 = get_img(path, f'{i}.bmp')
    #     p1 = p1.astype(np.float32) / 255
    #     p1 = torch.from_numpy(p1)
    #
    #     path = 'Data/test/Mask-ok'
    #     p2 = get_img(path, f'{i}.bmp')
    #     p2 = p2.astype(np.float32) / 255
    #     p2 = torch.from_numpy(p2)
    #
    #     cri = smp.utils.losses.DiceLoss()
    #     print(1 - cri(p1, p2).item())