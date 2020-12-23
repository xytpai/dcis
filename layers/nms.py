import torch
import torch.nn as nn 
import torch.nn.functional as F
import nms_cuda
from .misc import box_iou


def box_nms(bboxes, scores, threshold=0.5):
    '''
    Param:
    bboxes: FloatTensor(n,4) # 4: ymin, xmin, ymax, xmax
    scores: FloatTensor(n)

    Return:
    keep:   LongTensor(s)
    '''
    if bboxes.shape[0] == 0:
        return torch.zeros(0).long().to(bboxes.device)
    scores = scores.view(-1, 1)
    bboxes_scores = torch.cat([bboxes, scores], dim=1) # (n, 5)
    keep = nms_cuda.nms(bboxes_scores, threshold) # (s)
    return keep


def cluster_nms(classes, scores, boxes, nms_iou, other=[]):
    scores, idx = scores.sort(0, descending=True)
    boxes = boxes[idx]
    classes = classes[idx]
    mask = (classes.view(-1,1) == classes.view(1,-1)).float()
    iou = box_iou(boxes, boxes)*mask
    iou.triu_(diagonal=1)
    C = iou
    for i in range(200):
        A = C
        maxA = A.max(dim=0)[0]
        E = (maxA < nms_iou).float().unsqueeze(1).expand_as(A)
        C = iou.mul(E)
        if A.equal(C): break
    keep = maxA < nms_iou
    if len(other) > 0:
        other = [other[i][idx][keep] for i in range(len(other))]
    return classes[keep], scores[keep], boxes[keep], other
