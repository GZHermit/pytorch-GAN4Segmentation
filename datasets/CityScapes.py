# coding:utf-8
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

num_classes = 19
ignore_label = 255
id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label, 3: ignore_label,
                 4: ignore_label, 5: ignore_label, 6: ignore_label, 7: 0, 8: 1, 9: ignore_label,
                 10: ignore_label, 11: 2, 12: 3, 13: 4, 14: ignore_label, 15: ignore_label,
                 16: ignore_label, 17: 5, 18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                 25: 12, 26: 13, 27: 14, 28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17,
                 33: 18}


def make_datasets(quality, mode, root):
    '''

    :param quality: image quality,include 'fine' or 'coarse'
    :param mode: possible uses for the dataset, include 'train', 'val', or 'test'
    :param root: the root directory of the dataset.
    :return: items, consisted of tuples,
             every tuple consisted of an img_path and corresponding mask_path
    '''
    assert (quality == 'fine' and mode in ['train', 'val', 'test']) or \
           (quality == 'coarse' and mode in ['train', 'train_extra', 'val'])
    if quality == 'fine':
        img_dn = 'leftImg8bit'
        mask_dn = 'gtFine'
        img_postfix = '_leftImg8bit.png'
        mask_postfix = '_gtFine_labelIds.png'
    else:
        pass
    img_p = os.path.join(root, img_dn, mode)
    mask_p = os.path.join(root, mask_dn, mode)
    citys = os.listdir(img_p)
    items = []
    for c in citys:
        c_items = [n.split(img_postfix)[0] for n in os.listdir(os.path.join(img_p, c))]
        items += [(os.path.join(img_p, c, c_item + img_postfix),
                   os.path.join(mask_p, c, c_item + mask_postfix)) for c_item in c_items]
    return items


class CityScapes(Dataset):
    def __init__(self, mode, root=None, quality='fine'):
        self.imginfo = make_datasets(quality, mode, root)
        self.quality = quality
        self.mode = mode
        self.id_to_trainid = id_to_trainid

    def __getitem__(self, item):
        img_fp, mask_fp = self.imginfo[item]
        img, mask = Image.open(img_fp).convert('RGB'), Image.open(mask_fp)
        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))
        return mask

    def __len__(self):
        return len(self.imginfo)

# root = '/home/gzh/Workspace/Dataset/Cityscapes_DATA'
# cs = CityScapes(quality='fine', mode='val', root=root)
# i, m = cs[0]
# i = np.array(i)
# m = np.array(m)
# print(i.shape)
# print(m.shape)
