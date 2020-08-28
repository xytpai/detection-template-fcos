import torch 
import torch.nn as nn 
import torch.nn.functional as F 
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
        self.num_class    = self.cfg['DETECTOR']['NUM_CLASS']
        self.level_minmax = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 9999]]
        self.backbone   = ResNet(self.cfg['DETECTOR']['DEPTH'])
        self.neck       = FPN(self.backbone.out_channels, 256)
        self.head       = FCOSHead(256, self.num_class, norm=4.0)
        if self.mode == 'TRAIN' and self.cfg['TRAIN']['PRETRAINED']:
            self.backbone.load_pretrained_params()
        # loss 
        self.sigmoid_focal_loss = SigmoidFocalLoss(2.0, 0.25)
        self.iou_loss = IOULoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, imgs, locations, label_cls=None, label_reg=None):
        '''
        imgs:       F(b, 3, size, size)
        locations:  F(b, 5)
        label_cls:  L(b, n)       0:pad
        label_reg:  F(b, n, 4)
        '''
        batch_size, channels, im_h, im_w = imgs.shape
        out = self.neck(self.backbone(imgs))
        feature_size = []
        pred_cls, pred_reg, pred_ctr = [], [], []
        for s, feature in enumerate(out):
            h_s, w_s = feature.shape[2], feature.shape[3]
            cls_s, reg_s, ctr_s = self.head(feature, im_h, im_w)
            pred_cls.append(cls_s)
            pred_reg.append(reg_s)
            pred_ctr.append(ctr_s)
            feature_size.append([h_s, w_s])
        pred_cls = torch.cat(pred_cls, dim=1) # F(b, an, num_class)
        pred_reg = torch.cat(pred_reg, dim=1) # F(b, an, 4) yxyx
        pred_ctr = torch.cat(pred_ctr, dim=1) # F(b, an)
        if (label_cls is not None) and (label_reg is not None):
            return self._loss(locations, pred_cls, pred_reg, pred_ctr, feature_size, \
                                    im_h, im_w, label_cls, label_reg)
        else:
            return self._pred(locations, pred_cls, pred_reg, pred_ctr, feature_size, \
                                    im_h, im_w)
    
    def _loss(self, locations, pred_cls, pred_reg, pred_ctr, feature_size, 
                    im_h, im_w, label_cls, label_reg):
        loss = []
        for b in range(pred_cls.shape[0]):
            # filter out padding labels
            label_cls_b, label_reg_b = label_cls[b], label_reg[b]
            m = label_cls_b > 0
            label_cls_b = label_cls_b[m]
            label_reg_b = label_reg_b[m]
            # get target
            target_cls_b, target_reg_b, target_ctr_b = [], [], []
            for s in range(len(feature_size)):
                h_s, w_s = feature_size[s][0], feature_size[s][1]
                stride = (im_h-1) // (h_s-1)
                target_cls_s, target_reg_s, target_ctr_s, target_idx_s = \
                    assign_fcos(label_cls_b, label_reg_b, h_s, w_s, stride,
                            self.level_minmax[s][0], self.level_minmax[s][1])
                target_cls_b.append(target_cls_s)
                target_reg_b.append(target_reg_s)
                target_ctr_b.append(target_ctr_s)
            target_cls_b = torch.cat(target_cls_b, dim=0) # L(an)
            target_reg_b = torch.cat(target_reg_b, dim=0) # F(an, 4)
            target_ctr_b = torch.cat(target_ctr_b, dim=0) # F(an)
            # get loss
            m_negpos = target_cls_b > -1  # B(an)
            m_pos    = target_cls_b > 0   # B(an)
            num_pos = float(m_pos.sum().clamp(min=1))
            label_cls_selected = target_cls_b[m_negpos]
            label_reg_selected = target_reg_b[m_pos]
            label_ctr_selected = target_ctr_b[m_pos]
            pred_cls_selected = pred_cls[b][m_negpos] # F(n+-, num_class)
            pred_reg_selected = pred_reg[b][m_pos]    # F(n+, 4)
            pred_ctr_selected = pred_ctr[b][m_pos]    # F(n+)
            loss_cls = self.sigmoid_focal_loss(pred_cls_selected, label_cls_selected).view(1)
            loss_reg = self.iou_loss(pred_reg_selected, label_reg_selected).view(1)
            loss_ctr = self.bce_loss(pred_ctr_selected, label_ctr_selected).view(1)
            loss.append((loss_cls+loss_reg+loss_ctr)/num_pos)
        return torch.cat(loss)

    def _pred(self, locations, pred_cls, pred_reg, pred_ctr, feature_size, im_h, im_w):
        '''
        pred_cls_i: L(n)
        pred_cls_p: F(n)
        pred_reg:   F(n, 4)
        '''
        assert self.mode != 'TRAIN'
        batch_size = pred_cls.shape[0]
        assert batch_size == 1
        pred_cls = pred_cls.squeeze(0)
        pred_reg = pred_reg.squeeze(0)
        pred_ctr = pred_ctr.squeeze(0)
        pred_cls_p, pred_cls_i = torch.max(pred_cls.sigmoid(), dim=1)
        pred_cls_i = pred_cls_i + 1
        pred_cls_p = pred_cls_p * pred_ctr
        start = 0
        _pred_cls_i, _pred_cls_p, _pred_reg = [], [], []
        for size in feature_size:
            num = size[0]*size[1]
            p = pred_cls_p[start:start+num]
            nms_maxnum = min(int(self.cfg[self.mode]['NMS_TOPK_P']), num)
            select = torch.topk(p, nms_maxnum, largest=True, dim=0)[1]
            _pred_cls_i.append(pred_cls_i[start:start+num][select])
            _pred_cls_p.append(pred_cls_p[start:start+num][select])
            _pred_reg.append(pred_reg[start:start+num][select])
            start += num
        pred_cls_i = torch.cat(_pred_cls_i, dim=0)
        pred_cls_p = torch.cat(_pred_cls_p, dim=0)
        pred_reg = torch.cat(_pred_reg, dim=0)
        m = pred_cls_p > self.cfg[self.mode]['NMS_TH']
        pred_cls_i = pred_cls_i[m]
        pred_cls_p = pred_cls_p[m]
        pred_reg = pred_reg[m]
        pred_reg[:, 2].clamp_(max=locations[2])
        pred_reg[:, 3].clamp_(max=locations[3])
        pred_reg[:, 0::2] -= float(locations[0])
        pred_reg[:, 1::2] -= float(locations[1])
        pred_reg[:, :2].clamp_(min=0)
        pred_reg = pred_reg/float(locations[4])
        # nms for each class
        _pred_cls_i, _pred_cls_p, _pred_reg = [], [], []
        for cls_id in range(1, self.num_class+1):
            m = (pred_cls_i == cls_id).nonzero().view(-1)
            if m.shape[0] == 0: continue
            pred_cls_i_id = pred_cls_i[m]
            pred_cls_p_id = pred_cls_p[m]
            pred_reg_id = pred_reg[m]
            keep = box_nms(pred_reg_id, pred_cls_p_id, self.cfg[self.mode]['NMS_IOU'])
            _pred_cls_i.append(pred_cls_i_id[keep])
            _pred_cls_p.append(pred_cls_p_id[keep])
            _pred_reg.append(pred_reg_id[keep])
        return torch.cat(_pred_cls_i, dim=0), \
                    torch.cat(_pred_cls_p, dim=0), \
                        torch.cat(_pred_reg, dim=0)
