import torch
import torch.nn as nn
from time import time
import numpy as np
import os
import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Get Config yaml File")
    parser.add_argument("--root_path", required=False, default="/root")
    parser.add_argument("--config", required=True)
    return parser.parse_args()

def init_weights(layers):
    if isinstance(layers, nn.Conv2d):
        nn.init.kaiming_normal_(layers.weight)
    elif isinstance(layers, nn.Linear):
        nn.init.kaiming_normal_(layers.weight)
    elif isinstance(layers, nn.BatchNorm2d):
        nn.init.constant_(layers.weight, 1)
        nn.init.constant_(layers.bias, 0)

def to_device(data_dict, device):
    for k in data_dict.keys():
        data_dict[k] = data_dict[k].to(device=device)

def update_lr(optim, epoch, cfg, period=[10]):
    if epoch in period:
        for param_group in optim.param_groups:
            param_group['lr'] *= cfg.train.lr_decay

def cal_metric(predict, target):
    mask_pred = torch.gt(predict, 1e-2)
    mask_gt = torch.gt(target, 1e-2)
    mask = torch.logical_and(mask_gt, mask_pred)
    p = predict[mask]
    t = target[mask]
    diff = torch.abs(p - t)
    ratio = torch.max(p / t, t / p)

    delta1 = torch.sum( ratio < 1.25 ) / p.size(0) # Threshold Accuarcy 1.25
    delta2 = torch.sum( ratio < 1.25**2 ) / p.size(0) # Threshold Accuarcy 1.25^2
    delta3 = torch.sum( ratio < 1.25**3 ) / p.size(0) # Threshold Accuarcy 1.25^3
    RMS = torch.sqrt(torch.pow(diff, 2).mean()) # Root Mean Square Error
    Log = (torch.abs( torch.log10(p+1e-3) - torch.log10(t+1e-3) )).mean() # Averager log10 Error
    Rel = (diff / t).mean() # Relative Error
    SqRel = torch.sqrt(Rel) # Squared Relative Error

    return torch.tensor([delta1, delta2, delta3, RMS, Log, Rel, SqRel])

def show_metric(metrics):
    print(f"Delta_1: {metrics[0]:.3f} | Delta_2: {metrics[1]:.3f} | Delta_3: {metrics[2]:.3f} | RMS: {metrics[3]:.3f}| Log: {metrics[4]:.3f} | Rel: {metrics[5]:.3f} | SqRel: {metrics[6]:.3f}")

def save_model(args, model, DF):
    root_path = args.root_path
    name = str(int(time()))
    file_n = f"{root_path}/MDE_LiDAR/log/{name}"
    os.system(f"mkdir -p {file_n}")
    os.system(f"cp {root_path}/MDE_LiDAR/config/{args.config} {root_path}/MDE_LiDAR/log/{name}")
    torch.save(model.state_dict(), f"{root_path}/MDE_LiDAR/log/{name}/{name}.pth.tar")
    DF.to_csv(f"{args.root_path}/MDE_LiDAR/log/{name}/{name}.csv", index=False)