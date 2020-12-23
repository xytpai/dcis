import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import random
from layers import *
# TODO: choose backbone
from detectors.backbones import *
from detectors.necks import *
from detectors.heads import *


class Detector(nn.Module):
    def __init__(self, cfg, mode='TEST'):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.register_buffer('trained_log', torch.zeros(2).long())
        # ---
        self.num_class    = self.cfg['DETECTOR']['NUM_CLASS']
        self.alpha        = self.cfg['DETECTOR']['ALPHA']
        self.beta         = self.cfg['DETECTOR']['BETA']
        self.axis_range   = self.cfg['DETECTOR']['AXIS_RANGE']
        self.mask_th      = self.cfg[self.mode]['MASK_TH']
        self.class_th     = self.cfg[self.mode]['CLASS_TH']
        self.numdets      = 100
        # ---
        self.backbone     = ResNet(self.cfg['DETECTOR']['DEPTH'], 
                                use_dcn=self.cfg['DETECTOR']['USE_DCN'])
        self.neck         = FPN(self.backbone.out_channels, 256)
        # ---
        if self.mode == 'TRAIN' and self.cfg['TRAIN']['PRETRAINED']:
            self.backbone.load_pretrained_params()
        # conv_out
        self.conv_emb = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True))
        self.conv_out = nn.Conv2d(128, 
            self.num_class + self.axis_range*2, 
            kernel_size=1, padding=0)
        # loss_func
        self.loss_func_cls = nn.BCELoss(reduction='mean')
        self.loss_func_mask = nn.BCELoss(reduction='mean')
        
    def forward(self, imgs, locations, label_cls=None, label_reg=None, label_mask=None):
        '''
        imgs:       F(b, 3, im_h, im_w)
        locations:  F(b, 6)       yxyx ori_h ori_w
        label_cls:  L(b, n)       0:pad
        label_reg:  F(b, n, 4)    yxyx
        '''
        batch_size, _, im_h, im_w = imgs.shape
        bottom = self.neck(self.backbone(imgs))[0] # F(b, c, ph/2, pw/2)
        bottom = self.conv_emb(bottom)
        _, _, h_, w_ = bottom.shape
        bottom = F.interpolate(bottom, size=((h_-1)*2+1, (w_-1)*2+1)
            mode='bilinear', align_corners=True)
        bottom = self.conv_out(bottom)
        pred_cls, pred_y, pred_x = bottom.split(
            [self.num_class, self.axis_range, self.axis_range], dim=1)
        if label_cls is not None:
            return self._loss(locations, pred_cls, pred_y, pred_x, im_h, im_w, 
                label_cls, label_reg, label_mask)
        else:
            return self._pred(locations, pred_cls, pred_y, pred_x, im_h, im_w)
    
    def _loss(self, locations, pred_cls, pred_y, pred_x, im_h, im_w, 
                label_cls, label_reg, label_mask):
        loss = []
        batch_size, _, ph, pw = pred_cls.shape
        stride = (im_h-1)//(ph-1)
        ys, xs = torch.meshgrid(torch.arange(ph), torch.arange(pw), 
                    device=pred_cls.device, dtype=torch.float) # F(ph, pw)
        ys, xs = ys*stride, xs*stride
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
            # TODO: assign_centernet
            target = assign_centernet(label_cls_b-1, label_reg_b, 
                ph, pw, stride, self.num_class)
            loss_cls = self.loss_func_cls(pred_cls[b].sigmoid().view(-1), 
                target.view(-1)).view(1)
            # loss_mask
            ys_b = (ys.unsqueeze(0).expand(n, ph, pw) - \
                label_reg_ctr_b[:, 0].unsqueeze(1).unsqueeze(1))*self.alpha + self.beta
            xs_b = (xs.unsqueeze(0).expand(n, ph, pw) - \
                label_reg_ctr_b[:, 1].unsqueeze(1).unsqueeze(1))*self.alpha + self.beta
            mask_y = indexf2d(pred_y[b].sigmoid(), ys_b) # F(n, ph, pw)
            mask_x = indexf2d(pred_x[b].sigmoid(), xs_b) # F(n, ph, pw)
            pred_mask = mask_y*mask_x
            label_mask_b = F.interpolate(label_mask_b.unsqueeze(0), size=(ph, pw),
                mode='bilinear', align_corners=True)[0]
            loss_mask = self.loss_func_mask(pred_mask.view(-1), label_mask_b.view(-1)).view(1)
            # loss
            loss.append(loss_cls + loss_mask)
        return torch.cat(loss)

    def _pred(self, locations, pred_cls, pred_y, pred_x, im_h, im_w):
        '''
        cls:   L(n)
        score: F(n)
        mask:  F(n, ori_h, ori_w)
        '''
        assert self.mode != 'TRAIN'
        batch_size, _, ph, pw = pred_cls.shape
        assert batch_size == 1
        stride = (im_h-1)//(ph-1)
        ys, xs = torch.meshgrid(torch.arange(ph), torch.arange(pw), 
                    device=pred_cls.device, dtype=torch.float) # F(ph, pw)
        ys, xs = ys*stride, xs*stride
        # ---
        pred_cls = pred_cls[0].sigmoid()
        pred_cls = peakdet(pred_cls)
        # ---
        m = torch.where(pred_cls>=self.class_th, 
                torch.ones_like(pred_cls), torch.zeros_like(pred_cls))
        m_posi = m.max(dim=0)[0] # B(ph, pw)
        ctr_y = ys[m_posi]
        ctr_x = xs[m_posi]
        score, class_idx = pred_cls.max(dim=0)
        score = score[m_posi]
        class_idx = class_idx[m_posi]
        class_idx = class_idx + 1 # L(n)
        # ---
        ys_b = (ys.unsqueeze(0).expand(n, ph, pw) - \
            ctr_y.unsqueeze(1).unsqueeze(1))*self.alpha + self.beta
        xs_b = (xs.unsqueeze(0).expand(n, ph, pw) - \
            ctr_x.unsqueeze(1).unsqueeze(1))*self.alpha + self.beta
        # ---
        mask_y = indexf2d(pred_y[0].sigmoid(), ys_b) # F(n, ph, pw)
        mask_x = indexf2d(pred_x[0].sigmoid(), xs_b) # F(n, ph, pw)
        pred_mask = mask_y*mask_x
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
