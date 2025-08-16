import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms

def rotate(img, depth_img, mask, p=0.5, angle_range=(-30, 30), ignore_value=255):
    if random.random() < p:
        angle = random.uniform(angle_range[0], angle_range[1])        
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))        
        depth_img = depth_img.rotate(angle, resample=Image.NEAREST, fillcolor=0)
        mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=ignore_value)        
    return img, depth_img, mask

def crop(img, depth_img, mask, size, ignore_value=255):
    size_w, size_h = size[0], size[1]
    w, h = img.size
    padw = size_w - w if w < size_w else 0
    padh = size_h - h if h < size_h else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    depth_img = ImageOps.expand(depth_img, border=(0, 0, padw, padh), fill=0)  # 深度图像填充
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size_w)
    y = random.randint(0, h - size_h)
    img = img.crop((x, y, x + size_w, y + size_h))
    depth_img = depth_img.crop((x, y, x + size_w, y + size_h))
    mask = mask.crop((x, y, x + size_w, y + size_h))

    return img, depth_img, mask


def hflip(img, depth_img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        depth_img = depth_img.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转深度图像
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, depth_img, mask



def normalize(img, depth_img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    depth_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    ])(depth_img)
    # depth_img = transforms.ToTensor()(depth_img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, depth_img, mask
    return img, depth_img


def resize(img, depth_img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    depth_img = depth_img.resize((ow, oh), Image.NEAREST)  # 使用NEAREST邻近插值法调整深度图像大小
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, depth_img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size[1], img_size[0])
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size[0] * img_size[1]
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size[0])
        y = np.random.randint(0, img_size[1])

        if x + cutmix_w <= img_size[0] and y + cutmix_h <= img_size[1]:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask
