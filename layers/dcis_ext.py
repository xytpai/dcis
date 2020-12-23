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
    return detx_ext_cuda.peakdet(input)

