#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import random
import shutil
import time
import warnings
import pickle

from requests import patch

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main():

    # create model
    print("=> creating model '{}'".format("byol_Resnet50"))
    model = models.__dict__['resnet50']()

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()


    ckpt_path = "/root/code/ssl_dist/teacher/byol/pretrain_res50x1.pth.tar"

    state_dict = torch.load(ckpt_path)
    # model.load_state_dict(state_dict)
    for k in list(state_dict.keys()):
        if k.startswith('fc') :
            del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    for k in list(state_dict.keys()):
        print(k)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    print("done")

    # byol_model = torch.load(ckpt_path)
    # print(byol_model)
    # checkpoint = torch.load("/root/code/ssl_dist/teacher/byol/BYOL_pretrained_res50_solo.ckpt", map_location="cpu")
    # print(checkpoint)
    # state_dict = checkpoint['state_dict']
    # print(state_dict)

main()

        # rename distill pre-trained keys
    # for k in list(state_dict.keys()):
    #         print(k)
    #         # retain only encoder_q up to before the embedding layer
    #         if k.startswith('student') and not k.startswith('student.fc'):
    #             # remove prefix
    #             state_dict[k[len("student."):]] = state_dict[k]
    #         # delete renamed or unused k
    #         del state_dict[k]
    # else:
    #     if (args.pretrained_method == 'moco'):
    #         checkpoint = torch.load(args.pretrained, map_location="cpu")
    #         state_dict = checkpoint['state_dict']
    #         # rename moco pre-trained keys
    #         for k in list(state_dict.keys()):
    #             # retain only encoder_q up to before the embedding layer
    #             if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
    #                 # remove prefix
    #                 state_dict[k[len("module.encoder_q."):]] = state_dict[k]
    #             # delete renamed or unused k
    #             del state_dict[k]
    #     else:
    #         # pretrained_model = torch.load(args.pretrained, map_location="cpu")
    #         # state_dict = pretrained_model.state_dict()
    #         model_tmp = models.__dict__[args.arch]()
    #         model_tmp.load_state_dict (torch.load (args.pretrained, map_location="cpu"))
    #         state_dict = model_tmp.state_dict()

    #         for k in list(state_dict.keys()):
    #                 if k.startswith('fc') :
    #                     del state_dict[k]

    #         args.start_epoch = 0
    #         msg = model.load_state_dict(state_dict, strict=False)
    #         for k in list(state_dict.keys()):
    #             print(k)
    #         assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    #         print("=> loaded pre-trained model '{}'".format(args.pretrained))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.pretrained))



