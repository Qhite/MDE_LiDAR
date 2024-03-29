import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.loss import chamfer_distance

# Loss functions used in AdaBins paper

class Losses(nn.Module):
    def __init__(self, cfg):
        super(Losses, self).__init__()
        self.cfg = cfg
        self.SILL = SILogLoss()
        self.BCL = BinsChamferLoss()
        self.MML = MinMaxLoss()
    
    def forward(self, epoch, output, centers, d_dict):
        target = d_dict["depth"]
        if output.shape[-2:] != target.shape[-2:]:
            output = F.interpolate(output, [228, 304], mode="nearest")

        SIL_loss = self.SILL(output, target)
        BC_loss = self.BCL(centers, target)
        MM_loss = self.MML(centers, target)

        if epoch < 10:
            loss = self.cfg.train.alpha * SIL_loss + self.cfg.train.beta * BC_loss #+ self.cfg.train.gamma * MM_loss
        else:
            loss = self.cfg.train.alpha * SIL_loss + self.cfg.train.beta * BC_loss + self.cfg.train.gamma * MM_loss

        return loss

class SILogLoss(nn.Module):  
    def __init__(self, lamb=0.85):
        super(SILogLoss, self).__init__()
        self.name = 'SILogLoss'
        self.lamb = lamb

    def forward(self, output, target):
        mask_pred = output.ge(1e-2)
        mask_gt = target.ge(1e-2)
        mask = torch.logical_and(mask_pred, mask_gt)
        masked_output = output[mask]
        masked_target = target[mask]

        g = torch.log(masked_output + 1e-2) - torch.log(masked_target + 1e-2)

        Dg = torch.var(g) + (1 - self.lamb) * torch.pow(torch.mean(g), 2)
        losses = torch.sqrt(Dg)

        return losses

class BinsChamferLoss(nn.Module):
    def __init__(self):
        super(BinsChamferLoss, self).__init__()
        self.name = "ChamferLoss"
    
    def forward(self, bin_centers, target):
        if len(bin_centers.shape) == 1:
            bin_centers = bin_centers.unsqueeze(0).unsqueeze(2)
        else:
            bin_centers = bin_centers.unsqueeze(2)

        target_points = target.flatten(1)

        mask = target_points.ge(1e-2)
        target_points = [p[m] for p, m in zip(target_points, mask)]

        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target.device)

        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)

        loss, _ = chamfer_distance(x=bin_centers, y=target_points, y_lengths=target_lengths)

        return loss

class MinMaxLoss(nn.Module):
    def __init__(self):
        super(MinMaxLoss, self).__init__()
        self.name = "MinMaxLoss"
    
    def forward(self, centers, target):
        T = target.flatten(1)
        maxT = T.max(dim=1)[0]
        minT = T.min(dim=1)[0]
        return torch.abs(centers[:,-1] - maxT).sum() + torch.abs(centers[:,0] - minT).sum()