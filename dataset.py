# encoding: utf-8
"""
@author: FroyoZzz
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: froyorock@gmail.com
@software: garner
@file: dataset.py
@time: 2019-08-07 17:21
@desc:
"""
import torch
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


class CustomDataset(Dataset):
    def __init__(self, image_path = "data/TrainImages", mode = "train"):
        assert mode in ("train", "val", "test")
        self.image_path = image_path
        self.image_list = glob(os.path.join(self.image_path, "*.jpg"))
        self.mode = mode

        if mode in ("train", "val"):
            self.mask_path = self.image_path + "Masks"


    def __getitem__(self, index):
        if self.mode in ("train", "val"):
            image_name = self.image_list[index].split("\\")[-1].split(".")[0]
            X = Image.open(self.image_list[index])
            X = T.Resize((256, 256))(X)
            X = T.ToTensor()(X)
            mean = torch.mean(X)
            var = torch.var(X)
            std = torch.sqrt(var)
            X = T.Normalize([mean], [std])(X)

            mask = np.array(Image.open(os.path.join(self.mask_path, image_name+".jpg")).convert('1').resize((256, 256)))
            mask_rev = ~mask
            masks = np.empty([2, 256, 256], dtype=np.float64)
            masks[0] = mask
            masks[1] = mask_rev
            masks = torch.tensor(masks)

            # 以下是为了将mask变成两个矩阵, 一个全零
            # masks = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=np.uint8)
            # masks[:, :, 0] = mask
            # masks = T.ToTensor()(masks) * 255 # *255可以让它变成1，不知为何

            # mask = Image.open(os.path.join(self.mask_path, image_name + ".jpg")).convert('1')
            # mask = T.Resize((256, 256))(mask)
            # mask = T.ToTensor()(mask)

            return X, masks
        
        else:
            X = Image.open(self.image_list[index])
            X = T.Resize((256, 256))(X)
            X = T.ToTensor()(X)
            mean = torch.mean(X)
            var = torch.var(X)
            std = torch.sqrt(var)
            X = T.Normalize([mean], [std])(X)
            path = self.image_list[index]
            return X, path

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    dataset = CustomDataset()
    X, mask = dataset.__getitem__(0)
    X = X.numpy()
    mask = mask.numpy()
    # print(X)
    # print(mask[0])
    # print(mask.shape)