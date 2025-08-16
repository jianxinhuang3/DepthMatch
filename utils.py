import numpy as np
import logging
import os

import torch
from torch import nn
import torch.nn.functional as F
import cv2
import random
import matplotlib.pyplot as plt


class BoundaryLoss_1219(nn.Module):
    def __init__(self, threshold_depth=0.08, threshold_pred=0.1, use_dice=True):
        super(BoundaryLoss_1219, self).__init__()
        self.threshold_depth = threshold_depth
        self.threshold_pred = threshold_pred
        self.use_dice = use_dice
        
    def forward(self, pred_mask, depth):
        # depth: (B, H, W)
        # pred_mask: (B, H, W), predicted semantic mask or probability

        # 1. 提取深度边界
        depth_grad = self._compute_gradients(depth.float())  # (B,2,H,W), 分别为dx, dy
        depth_edge = torch.sqrt(depth_grad[:,0]**2 + depth_grad[:,1]**2)  
        B_gt = (depth_edge > self.threshold_depth).float()  # 二值化得到真实边界

        # 2. 提取预测掩码边界
        pred_grad = self._compute_gradients(pred_mask.float())
        pred_edge = torch.sqrt(pred_grad[:,0]**2 + pred_grad[:,1]**2)
        B_pred = (pred_edge > self.threshold_pred).float()
        # input(B_pred.max())
        B_gt = B_gt * B_pred

        # 3. 根据选择使用 BCE 或 Dice
        if self.use_dice:
            loss = self._dice_loss(B_pred, B_gt)
        else:
            # 使用二元交叉熵作为示例
            loss = F.binary_cross_entropy(B_pred, B_gt)
        
        return loss

    def _compute_gradients(self, x):
        # x: (B,H,W)
        # 使用Sobel或简单微分卷积提取梯度
        sobel_x = torch.tensor([[-1,0,1],
                                [-2,0,2],
                                [-1,0,1]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1,-2,-1],
                                [ 0, 0, 0],
                                [ 1, 2, 1]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)

        dx = F.conv2d(x.unsqueeze(1), sobel_x, padding=1)
        dy = F.conv2d(x.unsqueeze(1), sobel_y, padding=1)
        # dx, dy shape: (B,1,H,W)
        return torch.cat([dx, dy], dim=1)  # (B,2,H,W)

    def _dice_loss(self, input, target, eps=1e-6):
        # input: predicted edge (B,H,W)
        # target: ground truth edge (B,H,W)
        input_flat = input.view(input.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (input_flat * target_flat).sum(dim=1)
        dice = (2.*intersection + eps) / (input_flat.sum(dim=1) + target_flat.sum(dim=1) + eps)
        loss = 1 - dice.mean()
        return loss
   

class MaskGenerator:
    def __init__(self, input_size=[256, 320], mask_patch_size=32, model_patch_size=4, \
                 mask_ratio=0.6, mask_type='patch', strategy='comp'):
        self.input_size = np.array(input_size)
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        # 计算需要填充的大小，以便使input_size可以被mask_patch_size整除
        self.padded_size = np.ceil(self.input_size / self.mask_patch_size).astype(int) * self.mask_patch_size
        
        # 调整后的随机掩码尺寸
        self.rand_size = self.padded_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size[0] * self.rand_size[1]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

        if mask_type == 'patch':
            self.gen_mask = self.gen_patch_mask
        elif mask_type == 'square':
            self.gen_mask = self.gen_square_mask
        else:
            raise AssertionError("Not valid mask type!")

        if strategy == 'comp':
            self.strategy = self.gen_comp_masks
        elif strategy == 'rand_comp':
            self.strategy = self.gen_rand_comp_masks
        elif strategy == 'indiv':
            self.strategy = self.gen_indiv_masks
        else:
            raise AssertionError("Not valid strategy!")
 
    def gen_patch_mask(self):
        # 随机生成掩码索引
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        # 重新调整掩码的形状
        mask = mask.reshape((self.rand_size[0], self.rand_size[1]))
        mask = np.expand_dims(mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1), axis=0)
        
        # 裁剪掩码回原始大小
        mask = mask[:, :self.input_size[0], :self.input_size[1]]
        return mask

    def gen_square_mask(self):
        # 初始化一个全零掩码
        mask = np.zeros((self.padded_size[0], self.padded_size[1]), dtype=int)

        # 随机生成矩形掩码的起始位置和大小
        h1 = np.random.randint(0, self.input_size[0] * self.mask_ratio)
        w1 = np.random.randint(0, self.input_size[1] * self.mask_ratio)
        h2 = int(h1 + self.input_size[0] * self.mask_ratio)
        w2 = int(w1 + self.input_size[1] * self.mask_ratio)

        # 在掩码中填充矩形区域
        mask[w1:w2, h1:h2] = 1
        
        # 裁剪回原始大小
        mask = np.expand_dims(mask[:self.input_size[0], :self.input_size[1]], axis=0)
        return mask

    def gen_comp_masks(self):
        mask = self.gen_mask()
        return mask, 1 - mask

    def gen_rand_comp_masks(self):
        mask = self.gen_mask()
        nomask = np.zeros_like(mask)

        idx = random.randrange(3)
        if idx == 0:   return nomask, 1 - mask
        elif idx == 1: return mask, nomask
        elif idx == 2: return mask, 1 - mask

    def gen_indiv_masks(self):
        mask1 = self.gen_mask()
        mask2 = self.gen_mask()
        return mask1, mask2

    def __call__(self):
        return self.strategy()

        

# 可视化RGB图像
def save_rgb_image(rgb_tensors, filename="rgb.jpg"):
    for rgb_tensor in rgb_tensors:
        # 确保张量是 (3, H, W) 形状，使用 permute 将其转换为 (H, W, 3)
        rgb_image = rgb_tensor.permute(1, 2, 0).cpu().numpy()  # 如果在 GPU 上，则需要 .cpu()
        input(rgb_image.max())
        input(rgb_image.min())


        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

        if rgb_image.max() > 1.0:
            rgb_image = rgb_image / 255.0
        # 使用 matplotlib 的 imsave 保存图像为 .jpg 文件
        plt.imsave(filename, rgb_image)

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    return cmap


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explicit requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
