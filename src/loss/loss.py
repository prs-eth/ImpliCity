# encoding: utf-8
# Author: Bingxin Ke
# Created: 2021/11/16
import torch.nn as nn

"""
Wrap loss functions due to different input dtype requirements.

"""

ce = nn.CrossEntropyLoss(reduction='none')

bce_logits = nn.BCEWithLogitsLoss(reduction='none')


def wrapped_cross_entropy(pred, gt):
    return ce(pred, gt.long())


def wrapped_bce(pred, gt):
    return bce_logits(pred, gt.float())
