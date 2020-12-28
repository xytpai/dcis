import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import random, math
from layers import *
# TODO: choose backbone
from detectors.backbones import *
from detectors.necks import *


class Detector(nn.Module):
    def __init__(self, cfg, mode='TEST'):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.register_buffer('trained_log', torch.zeros(2).long())
        self.num_class  = self.cfg['DETECTOR']['NUM_CLASS']
        self.emb_size   = self.cfg['DETECTOR']['EMB_SIZE']
        self.numdets    = self.cfg['DETECTOR']['NUMDETS']
        self.backbone   = ResNet(self.cfg['DETECTOR']['DEPTH'], 
                            use_dcn=self.cfg['DETECTOR']['USE_DCN'])
        self.neck       = FPN(self.backbone.out_channels, 256, 
                            top_mode=None)
        if self.mode == 'TRAIN' and self.cfg['TRAIN']['PRETRAINED']:
            self.backbone.load_pretrained_params()
        # head
        self.conv_emb = nn.Sequential(
            nn.Conv2d(256 + 2, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True))
        self.conv_cls = nn.Conv2d(256, self.num_class, kernel_size=3, padding=1)
        self.conv_key = nn.Conv2d(256, self.emb_size, kernel_size=3, padding=1)
        self.conv_val = nn.Conv2d(256, self.emb_size, kernel_size=3, padding=1)
        # init conv_cls
        pi = 0.01
        _bias = -math.log((1.0-pi)/pi)
        nn.init.constant_(self.conv_cls.bias, _bias)
        # metric
        self.conv_metric = nn.Sequential(
            nn.Conv2d(self.emb_size, self.emb_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.emb_size, 1, kernel_size=3, padding=1))
        # loss_func
        self.loss_func_cls = neg_loss
        self.loss_func_mask = dice_loss
        
    def forward(self, imgs, locations, label_cls=None, label_reg=None, label_mask=None):
        '''
        imgs:       F(b, 3, im_h, im_w)
        locations:  F(b, 6)       yxyx ori_h ori_w
        label_cls:  L(b, n)       0:pad
        label_reg:  F(b, n, 4)    yxyx
        '''
        batch_size, _, im_h, im_w = imgs.shape
        res = self.neck(self.backbone(imgs))
        bottom = res[0] + res[1] + res[2]
        _, _, bh, bw = bottom.shape
        # compute coord
        ys, xs = torch.meshgrid(
            torch.arange(bh, device=bottom.device), 
            torch.arange(bw, device=bottom.device))
        ys = ys.float() / bh * 2.0 - 1.0
        xs = xs.float() / bw * 2.0 - 1.0
        coord = torch.stack([ys, xs]).unsqueeze(0).expand(batch_size, 2, bh, bw)
        bottom = torch.cat([bottom, coord], dim=1)
        bottom = self.conv_emb(bottom)
        _, _, bh, bw = bottom.shape
        bottom = F.interpolate(bottom, size=((bh-1)*2+1, (bw-1)*2+1), 
            mode='bilinear', align_corners=True)
        pred_cls = self.conv_cls(bottom)
        pred_key = self.conv_key(bottom)
        pred_val = self.conv_val(bottom)
        if label_cls is not None:
            return self._loss(locations, pred_cls, pred_key, pred_val, im_h, im_w, 
                label_cls, label_reg, label_mask)
        else:
            return self._pred(locations, pred_cls, pred_key, pred_val, im_h, im_w)
    
    def _loss(self, locations, pred_cls, pred_key, pred_val, im_h, im_w, 
                label_cls, label_reg, label_mask):
        loss = []
        batch_size, _, ph, pw = pred_cls.shape
        stride = (im_h-1)//(ph-1)
        ys, xs = torch.meshgrid(
            torch.arange(ph, device=pred_cls.device), 
            torch.arange(pw, device=pred_cls.device)) # L(ph, pw)
        for b in range(batch_size):
            # filter out padding labels
            label_cls_b, label_reg_b, label_mask_b = \
                label_cls[b], label_reg[b], label_mask[b]
            m = label_cls_b > 0
            label_cls_b = label_cls_b[m] # L(n) 1~80
            label_reg_b = label_reg_b[m] # F(n, 4)
            label_mask_b = label_mask_b[m] # F(n, im_h, im_w)
            label_reg_ctr_b = (label_reg_b[:, :2] + label_reg_b[:, 2:])/2.0 # F(n, 2)
            n = label_cls_b.shape[0]
            # loss_cls
            target = assign_centernet(label_cls_b-1, label_reg_b, 
                ph, pw, stride, self.num_class)
            loss_cls = self.loss_func_cls(pred_cls[b].view(-1).sigmoid(), 
                target.view(-1)).view(1)
            # loss_mask
            ctr = (label_reg_ctr_b[:, :]/stride).long() # L(n, 2)
            idxs = ctr[:,0] * pw + ctr[:,1] # L(n)
            key = pred_key[b].permute(1,2,0).contiguous() # F(ph, pw, c)
            val = pred_val[b].permute(1,2,0).contiguous() # F(ph, pw, c)
            key = key.view(ph*pw, -1)[idxs] # F(n, c)
            pred_mask = (key.view(n, 1, 1, self.emb_size) - \
                val.view(1, ph, pw, self.emb_size)).abs() # F(n, ph, pw, c)
            pred_mask = self.conv_metric(pred_mask.permute(0, 3, 1, 2).contiguous() \
                )[:, 0].sigmoid()
            label_mask_b = F.interpolate(label_mask_b.unsqueeze(0), size=(ph, pw),
                mode='bilinear', align_corners=True)[0]
            loss_mask = self.loss_func_mask(pred_mask, label_mask_b).view(1)
            # loss
            loss.append(loss_cls + loss_mask)
        return torch.cat(loss)

    def _pred(self, locations, pred_cls, pred_key, pred_val, im_h, im_w):
        '''
        cls:   L(n)
        score: F(n)
        mask:  F(n, ori_h, ori_w)
        '''
        assert self.mode != 'TRAIN'
        self.mask_th      = self.cfg[self.mode]['MASK_TH']
        self.class_th     = self.cfg[self.mode]['CLASS_TH']
        batch_size, _, ph, pw = pred_cls.shape
        assert batch_size == 1
        # to class_idx, score
        pred_cls = pred_cls[0].sigmoid()
        pred_cls = peakdet(pred_cls)
        m = torch.where(pred_cls>=self.class_th, 
                torch.ones_like(pred_cls), torch.zeros_like(pred_cls))
        m_posi = (m.max(dim=0)[0]).byte() # B(ph, pw)
        score, class_idx = pred_cls.max(dim=0)
        score = score[m_posi]
        class_idx = class_idx[m_posi]
        class_idx = class_idx + 1 # L(n)
        n = class_idx.shape[0]
        # to mask
        stride = (im_h-1)//(ph-1)
        ys, xs = torch.meshgrid(
            torch.arange(ph, device=class_idx.device), 
            torch.arange(pw, device=class_idx.device)) # F(ph, pw)
        ctr_ys = ys[m_posi]
        ctr_xs = xs[m_posi]
        idxs = ctr_ys * pw + ctr_xs
        key = pred_key[0].permute(1,2,0).contiguous() # F(ph, pw, c)
        val = pred_val[0].permute(1,2,0).contiguous() # F(ph, pw, c)
        key = key.view(ph*pw, -1)[idxs] # F(n, c)
        pred_mask = (key.view(n, 1, 1, self.emb_size) - \
            val.view(1, ph, pw, self.emb_size)).abs() # F(n, ph, pw, c)
        pred_mask = self.conv_metric(pred_mask.permute(0,3,1,2).contiguous() \
            )[:, 0].sigmoid()
        # post
        pred_mask = F.interpolate(pred_mask.unsqueeze(0), size=(im_h, im_w),
            mode='bilinear', align_corners=True)
        valid_ymin, valid_xmin, valid_ymax, valid_xmax, ori_h, ori_w = \
            float(locations[0]), float(locations[1]), \
            float(locations[2]), float(locations[3]), \
            int(locations[4]), int(locations[5])
        pred_mask = pred_mask[:, :, int(valid_ymin):int(valid_ymax)+1, 
                        int(valid_xmin):int(valid_xmax)+1]
        pred_mask = F.interpolate(pred_mask, size=(ori_h, ori_w), 
                    mode='bilinear', align_corners=True)[0]
        pred_mask = (pred_mask>=self.mask_th).float()
        return class_idx, score, pred_mask
