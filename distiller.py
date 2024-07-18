# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F


class Distiller(nn.Module):
    def __init__(self, base_encoder, teacher, args, h_dim=512):
        super(Distiller, self).__init__()
        self.args = args

        # create the encoders
        # num_classes is the output fc dimension
        self.teacher = teacher

        if 'efficient' in args.arch:
            self.student = base_encoder
        else:
            self.student = base_encoder()

        s_dim = self.student.fc.weight.shape[1]
        t_dim = self.teacher.fc.weight.shape[1]


        # baseline : encoder만 ->feature vector -> dimension matching만 해준다
        # exp1: data aug 없이
        # exp2: projection head 추가

        self.predictor = nn.Sequential(
            nn.Linear(s_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, t_dim)
        )

        self.projector = nn.Sequential(
            nn.Linear (t_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, s_dim)
        )

    def forward(self, x1, x2, feature_reduction=False):
        """
        x1 : first view of x
        x2 : second view of x
        """

        if (feature_reduction):
            p1, p2 = self.student(x1, is_feat=True), self.student(x2, is_feat=True)
            z1, z2 = self.teacher(x1, is_feat=True), self.teacher(x2, is_feat=True)
            t1, t2 = self.projector(z1), self.projector(z2)

        else:
            z1, z2 = self.student(x1, is_feat=True), self.student(x2, is_feat=True)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            t1, t2 = self.teacher(x1, is_feat=True), self.teacher(x2, is_feat=True)

        

        return p1, p2, t1, t2


def D(p, z, weight=None, version='simplified'): # negative cosine similarity
    # weight : weights for weighted sum with size [batch]

    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        if weight is None:
            return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
        else:
            loss = - F.cosine_similarity(p, z.detach(), dim=-1) # [batch]
            return (weight * loss).mean()