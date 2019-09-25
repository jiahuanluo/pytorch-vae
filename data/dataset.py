# -*- coding:utf-8 -*-
# @Author   : LuoJiahuan
# @File     : dataset.py 
# @Time     : 2019/9/25 17:16


from PIL import Image
from torch.utils.data import Dataset
import os


class MyDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        imgs = [os.path.join(dir_path, img_name) for img_name in os.listdir(dir_path)]
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, item):
        img = self.imgs[item]
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)
