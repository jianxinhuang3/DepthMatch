from dataset.transform_d import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        
        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None and nsample > len(self.ids):
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')   
        depth_img = Image.open(os.path.join(self.root, id.split(' ')[1])).convert('L').convert('RGB')    

        if self.mode == 'train_u':
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        else:
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[2])))) 
        
        if self.mode == 'val':
            img, depth_img, mask = normalize(img, depth_img, mask)
            mask[mask == 0] = 255
            mask = mask - 1
            mask[mask == 254] = 255
            return img, depth_img, mask, id

        img, depth_img, mask = resize(img, depth_img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, depth_img, mask = rotate(img, depth_img, mask, p=0.5, angle_range=(-30, 30), ignore_value=ignore_value)
        img, depth_img, mask = crop(img, depth_img, mask, self.size, ignore_value)
        img, depth_img, mask = hflip(img, depth_img, mask, p=0.5)

        if self.mode == 'train_l':      
            img, depth_img, mask = normalize(img, depth_img, mask)
            mask[mask == 0] = 255
            mask = mask - 1
            mask[mask == 254] = 255
            return img, depth_img, mask
        
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size, p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size, p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, dimg_s1, ignore_mask = normalize(img_s1, depth_img, ignore_mask)
        img_s2, dimg_s2 = normalize(img_s2, depth_img)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        img_w, dimg_w = normalize(img_w, depth_img)

        return img_w, dimg_w, img_s1, dimg_s1, img_s2, dimg_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
