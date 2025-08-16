import argparse
from copy import deepcopy
import logging
import os
import pprint
import numpy as np
from PIL import Image
import random

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi_d import SemiDataset
from model.semseg.dpt_dprompt import DPT
from evaluate import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed

from utils import MaskGenerator
from utils import BoundaryLoss_1219 as loss_BL

parser = argparse.ArgumentParser(description='DepthMatch: Semi-Supervised RGB-D Scene Parsing through Depth-Guided Regularization')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    seed = 114514
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass'], 'img_scale': cfg['crop_size']})
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    # model.backbone.load_state_dict(state_dict)
    model.backbone.load_state_dict(state_dict, strict=False)
        
    if cfg['lock_backbone']:
        model.lock_backbone()

    # model.backbone.cls_token_ada.data.copy_(model.backbone.cls_token.data)
    # model.backbone.pos_embed_ada.data.copy_(model.backbone.pos_embed.data)
    # model.backbone.mask_token_ada.data.copy_(model.backbone.mask_token.data)
    # model.backbone.patch_embed_ada.proj.weight.data.copy_(model.backbone.patch_embed.proj.weight.data)
    # model.backbone.patch_embed_ada.proj.bias.data.copy_(model.backbone.patch_embed.proj.bias.data)

    exclude_block_names = ['prompt','ada']
    # for name, param in model.backbone.named_parameters():
    #     if not any(block_name in name for block_name in exclude_block_names):
    #         param.requires_grad = False
    optimizer = AdamW(
        [
            {'params': [param for name, param in model.backbone.named_parameters() 
                        if not any(block_name in name for block_name in exclude_block_names)], 'lr': cfg['lr']},
            {'params': [param for name, param in model.backbone.named_parameters() 
                        if any(block_name in name for block_name in exclude_block_names)], 'lr': cfg['lr'] * cfg['lr_multi']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    
    # # 打印 backbone 中每个参数的学习率
    # for name, param in model.named_parameters():
    #     if 'backbone' in name:
    #         # lr = next((group['lr'] for group in optimizer.param_groups if param in group['params']), None)
    #         lr = next((group['lr'] for group in optimizer.param_groups if any(torch.equal(param, p) for p in group['params'])), None)
    #         print(f"Parameter: {name}, Learning Rate: {lr}")

    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    
    criterion_u_d = loss_BL().cuda(local_rank)

    trainset_u = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    )
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val'
    )
    
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l
    )
    
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u
    )
    
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
    )
    
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))
        
        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()
        total_loss_u_d = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        
        model.train()

        mask_generator = MaskGenerator(input_size=[cfg['crop_size'][1], cfg['crop_size'][0]], mask_patch_size=32, model_patch_size=1, mask_ratio=0.1, mask_type='patch', strategy='comp')
        
        for i, ((img_x, depth_x, mask_x),
                (img_u_w, depth_u_w, img_u_s1, dimg_u_s1, img_u_s2, dimg_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):                
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
            depth_x, depth_u_w, dimg_u_s1, dimg_u_s2 = depth_x.cuda(), depth_u_w.cuda(), dimg_u_s1.cuda(), dimg_u_s2.cuda()

            with torch.no_grad():
                pred_u_w = model_ema([img_u_w,depth_u_w]).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)
            
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            dimg_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = dimg_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]

            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            dimg_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = dimg_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]            
            
            mix_mask1, _ = mask_generator()
            mix_mask1 = torch.from_numpy(mix_mask1)

            img_u_s1_t = img_u_s1
            img_u_s2_t = img_u_s2
            img_u_s1[mix_mask1.expand(img_u_s1.shape) == 1] = dimg_u_s1[mix_mask1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[mix_mask1.expand(img_u_s2.shape) == 1] = dimg_u_s2[mix_mask1.unsqueeze(1).expand(img_u_s2.shape) == 1]
            dimg_u_s1[mix_mask1.expand(img_u_s1.shape) == 1] = img_u_s1_t[mix_mask1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            dimg_u_s2[mix_mask1.expand(img_u_s2.shape) == 1] = img_u_s2_t[mix_mask1.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            pred_x = model([img_x, depth_x])
            pred_u_s1, pred_u_s2 = model([torch.cat((img_u_s1, img_u_s2)), torch.cat((dimg_u_s1, dimg_u_s2))], comp_drop=True).chunk(2)
            
            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]
            
            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]
            
            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()
            
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()
            
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0

            ft = 1e-3 * (cfg['epochs'] - epoch)
            pred_u_s1 = pred_u_s1.argmax(dim=1)
            dimg_u_s1_c1 = dimg_u_s1.select(1, 0)  # B,H,W
            loss_u_d_s1 = criterion_u_d(pred_u_s1, dimg_u_s1_c1) * ft

            pred_u_s2 = pred_u_s2.argmax(dim=1)
            dimg_u_s2_c2 = dimg_u_s2.select(1, 0)  # B,H,W
            loss_u_d_s2 = criterion_u_d(pred_u_s2, dimg_u_s2_c2) * ft
            loss_u_d = (loss_u_d_s1 + loss_u_d_s2) / 2.0
            
            loss = (loss_x + loss_u_s) / 2.0 + loss_u_d
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())
            total_loss_u_d.update(loss_u_d.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
                writer.add_scalar('train/loss_d', loss_u_d.item(), iters)

            if (i % (len(trainloader_u) // 20) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss d: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg, 
                                            total_loss_s.avg, total_loss_u_d.avg, total_mask_ratio.avg))
        
        eval_mode = 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)
        mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=14)
        mIoU_slid, iou_class_slid = mIoU, iou_class
        
        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                            'EMA: {:.2f}, IoU_slid: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou, iou_class_ema[cls_idx], iou_class_slid[cls_idx]))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}, MIoU_slid: {:.2f}\n'.format(eval_mode, mIoU, mIoU_ema, mIoU_slid))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_ema', mIoU_ema, epoch)
            writer.add_scalar('eval/mIoU_slid', mIoU_slid, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                writer.add_scalar('eval/%s_IoU_ema' % (CLASSES[cfg['dataset']][i]), iou_class_ema[i], epoch)
                writer.add_scalar('eval/%s_IoU_slid' % (CLASSES[cfg['dataset']][i]), iou_class_slid[i], epoch)

        is_best = mIoU >= previous_best
        
        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'previous_best_ema': previous_best_ema,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
