import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import dcis_ext_cuda
import math


def peakdet(input):
    c, h, w = input.shape
    assert input.is_cuda
    assert input.dtype == torch.float
    return dcis_ext_cuda.peakdet(input)


def assign_centernet(cls_idx, bbox, ph, pw, stride, num_class):
    # cls_idx: L(n)
    # bbox: F(n, 4)
    assert cls_idx.is_cuda
    assert bbox.is_cuda
    assert cls_idx.dtype == torch.long
    assert bbox.dtype == torch.float
    n = cls_idx.shape[0]
    assert bbox.shape == (n, 4)
    cls_idx = cls_idx.view(1,1,n).expand(ph,pw,n)
    bbox = bbox.view(1,1,n,4).expand(ph,pw,n,4)
    return dcis_ext_cuda.assign_centernet(cls_idx, bbox, stride, num_class)
