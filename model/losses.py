from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCLoss(nn.Module):
    def __init__(self, gamma=1):
        super(PCLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.gamma = gamma

    def forward(self, z1, z2):
        cos_ = self.cos(z1, z2.detach())
        loss = torch.mul(torch.pow((2 - cos_), self.gamma), cos_)
        L = -loss.mean() + 1
        return L


class KLloss(nn.Module):
    def __init__(self):
        super(KLloss, self).__init__()

    def forward(self, z1, z2):
        kl = F.kl_div(z1.softmax(dim=-1).log(), z2.softmax(dim=-1), reduction='mean')

        return kl


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss
