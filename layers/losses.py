import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable


def neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    loss = 0
    pos_loss = torch.log(pred + 1e-10) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred + 1e-10) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def dice_loss(input, target):
    # input: F(n, -1)
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return (1-d).mean()


def binary_focal_loss(input, one_hot, reduction='sum'):
    gamma = 2
    alpha = 0.25
    pt = input*one_hot + (1.0-input)*(1.0-one_hot)
    w = alpha*one_hot + (1.0-alpha)*(1.0-one_hot)
    w = w * torch.pow((1.0-pt), gamma)
    loss = -w * pt.log()
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    else:
        return loss
