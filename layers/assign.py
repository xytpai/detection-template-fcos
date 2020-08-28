import torch
import torch.nn as nn 
import torch.nn.functional as F
import assign_cuda


def assign_fcos(label_cls, label_box, ph, pw, stride, size_min, size_max):
    '''
    Param:
    label_cls: L(n) 
    label_box: L(n, 4) # ymin, xmin, ymax, xmax

    Return:
    target_cls: L(ph*pw)
    target_reg: F(ph*pw, 4)
    target_ctr: F(ph*pw)
    target_idx: L(ph*pw)
    '''
    assert label_cls.is_cuda
    assert label_box.is_cuda
    n = label_cls.shape[0]
    assert n == label_box.shape[0]
    assert label_box.shape[1] == 4

    label_cls = label_cls.view(1, n).expand(ph*pw, n)
    label_box = label_box.view(1, n, 4).expand(ph*pw, n, 4)

    output = assign_cuda.assign_fcos(label_cls, label_box, ph, pw, stride, size_min, size_max)
    target_cls, target_reg, target_ctr, target_idx = output.split([1,4,1,1], dim=1)
    target_cls = target_cls.squeeze(1).long()
    target_ctr = target_ctr.squeeze(1)
    target_idx = target_idx.squeeze(1).long()
    return target_cls, target_reg, target_ctr, target_idx
