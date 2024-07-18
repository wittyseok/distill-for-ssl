#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import moco.loader

from distiller import Distiller, D
from models import model_dict
from tensorboardX import SummaryWriter
from models.efficientnet import EfficientNet
from models.mobilenetv3 import mobilenetv3_large

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

model_names = sorted(name for name in model_dict
    if name.islower() and not name.startswith("__")
    and callable(model_dict[name]))

model_names += ['efficientnet-b0', 'efficientnet-b1', 'mobilenetv3-large', 'efficientnet-b3']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# dataset directory
parser.add_argument('--data', metavar='DIR', default='/root/code/data/imagenet_100',
                    help='path to dataset')
# dataset selection
parser.add_argument('--data_type', metavar='DT', default='imagenet', 
                    help='choose type of data provided by pytorch')
# model selection
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
# gpu worker
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
# epoch
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
# when restarting, manually select epoch number
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# batch size
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node')
# learning rate
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
# schedule
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
# momentum
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
# weight decay
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
# print frequency
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# save frequency
parser.add_argument('--save-freq', default=10, type=int,
                    help='save frequency (default: 10)')
parser.add_argument('--log-freq', default=200, type=int,
                    help='log frequency (default: 200)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# options for moco v2
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# distillation options
parser.add_argument('--teacher', type=str, default='/root/code/ssl_dist/teacher/byol/BYOL_pretrained_new.pth',
                    help='path to pretrained teacher (which was trained by BYOL)')
parser.add_argument('--teacher_method', type=str, default='byol',
                    help='method the teacher was trained')
parser.add_argument('--t_arch', metavar='T_ARCH', default='resnet50',
                    choices=model_names,
                    help='teacher model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--pred_h_dim', type=int, default=512,
                    help='predictor hidden dimenstion')

# warm-up
parser.add_argument('--warm', action='store_true', help='use warm up lr')
parser.add_argument('--warm_epochs', type=int, default=5,
                    help='warm-up epochs')

# experiment 1 (without data augmentation)
parser.add_argument('--no_augmentation', action='store_true', 
                    help='no image augmentation')

# experiment 2 (with projection head)
parser.add_argument('--feature_reduction', action='store_true', 
                    help='reduce feature vector with projection head')

def main():
    args = parser.parse_args()

    args.model_path = './save/distill/models'
    args.tb_path = './save/distill/tensorboards'

    if (args.no_augmentation and args.feature_reduction):
        args.model_name = '{}_{}_lr_{}_bsz{}_pred_h_dim_{}_t_{}_no_aug_ftr_red'.format(
            args.data_type, args.arch, args.lr, args.batch_size, args.pred_h_dim, args.teacher_method
        )
    else :
        if (args.no_augmentation):
            args.model_name = '{}_{}_lr_{}_bsz{}_pred_h_dim_{}_t_{}_no_aug'.format(
                args.data_type, args.arch, args.lr, args.batch_size, args.pred_h_dim, args.teacher_method
            )
        elif (args.feature_reduction):
            args.model_name = '{}_{}_lr_{}_bsz{}_pred_h_dim_{}_t_{}_ftr_red'.format(
                args.data_type, args.arch, args.lr, args.batch_size, args.pred_h_dim, args.teacher_method
            )
        else:
            args.model_name = '{}_{}_lr_{}_bsz{}_pred_h_dim_{}_t_{}'.format(
                args.data_type, args.arch, args.lr, args.batch_size, args.pred_h_dim, args.teacher_method
            )


    args.tb_folder = os.path.join(args.tb_path, args.model_name)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)
    
    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    args.cos = True
    args.warm = True
    args.tb_steps = 0

    if args.warm:
        args.warmup_to = 0.5 * (1. + math.cos(math.pi * args.warm_epochs / args.epochs))
        args.warmup_from = 0

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))


    # load teacher
    print("==> loading teacher '{}'".format(args.teacher))

    teacher = model_dict[args.t_arch]()
    for name, param in teacher.named_parameters():
        param.requires_grad = False

    # load byol
    if (args.teacher_method == 'byol'):
        state_dict = torch.load(args.teacher, map_location="cpu")
        for k in list(state_dict.keys()):
                if k.startswith('fc') :
                    del state_dict[k]


    # load moco v2
    else : 
        checkpoint = torch.load(args.teacher, map_location="cpu")
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            print(k)
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
            
    args.start_epoch = 0
    msg = teacher.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    print("==> done")

    # create model
    print("==> creating model '{}'".format(args.arch))
    if not args.arch in ['efficientnet-b0', 'efficientnet-b1', 'mobilenetv3-large']:
        model = Distiller(
            base_encoder=model_dict[args.arch],
            teacher=teacher, h_dim=args.pred_h_dim, args=args)
    
    elif 'efficientnet' in args.arch:
        model = Distiller(
            base_encoder=EfficientNet.from_name(args.arch),
            teacher=teacher, h_dim=args.pred_h_dim, args=args)
    else:
        # mobile net v3 - large
        model = Distiller(
            base_encoder=mobilenetv3_large(),
            teacher=teacher, h_dim=args.pred_h_dim, args=args)
    print("==> done")

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    if (args.data_type == 'cifar10'):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.247, 0.243, 0.261])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])  
   
    if args.no_augmentation:
        transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
        ])

    else:
       # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        # apply random transformation t1, t2 each
        transform = moco.loader.TwoCropsTransform (transforms.Compose(augmentation)) 

    # load data
    if (args.data_type == 'cifar10'):
        print ('===> cifar10 data load')
        train_dataset =  datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    else:
        print ('===> imagenet data load')
        traindir = os.path.join(args.data, 'train')
        train_dataset = datasets.ImageFolder(traindir, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    logger = SummaryWriter(logdir=args.tb_folder)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, optimizer, epoch+1, args, logger)

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()}, 
            filename=os.path.join(args.save_folder, 'checkpoint_{:04d}epoch.pth.tar'.format(epoch+1)))
            
            # for resume
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()}, 
            filename=os.path.join(args.save_folder, 'latest.pth.tar'))


def train(train_loader, model, optimizer, epoch, args, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            if args.no_augmentation:
                images = images.cuda(args.gpu, non_blocking=True)

            else:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # warm-up learning rate
        warmup_learning_rate(args, epoch, i, len(train_loader), optimizer)

        # compute output
        if args.no_augmentation and args.feature_reduction:
            p, _, t, _ = model(x1=images, x2=images, feature_reduction=True)
            loss = D(p, t)
            losses.update(loss.item(), images.size(0))
        
        else:
            if args.no_augmentation:           
                p, _, t, _ = model(x1=images, x2=images)
                loss = D(p, t)
                losses.update(loss.item(), images.size(0))

            elif args.feature_reduction:           
                p1, p2, t1, t2 = model(x1=images[0], x2=images[1], feature_reduction=True)
                loss = D(p1, t1) / 2 + D(p2, t2) / 2
                losses.update(loss.item(), images[0].size(0))      

            else:
                p1, p2, t1, t2 = model(x1=images[0], x2=images[1])
                loss = D(p1, t1) / 2 + D(p2, t2) / 2
                losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if i % args.log_freq == 0:
            args.tb_steps += 1
            logger.add_scalar('Train Loss', loss.item(), args.tb_steps)
            logger.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], args.tb_steps)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
