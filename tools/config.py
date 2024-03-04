import os
import yaml
from yacs.config import CfgNode as CN

def get_attr(file_name):
    with open(file_name) as f:
        fyaml = yaml.load(f, Loader=yaml.FullLoader)
    
    cfg = CN()
    cfg.device = fyaml["device"]

    cfg.data = CN()
    cfg.data.image_size = fyaml["data"]["image_size"]
    cfg.data.lidar_size = fyaml["data"]["lidar_size"]

    cfg.model = CN()
    cfg.model.backbone = fyaml["model"]["backbone"]
    cfg.model.bin_size = fyaml["model"]["bin_size"]
    cfg.model.decoder_type = fyaml["model"]["decoder_type"]
    cfg.model.decoder_dim = fyaml["model"]["decoder_dim"]
    cfg.model.num_head = fyaml["model"]["num_head"]
    cfg.model.d_model = fyaml["model"]["emb_dim"]

    cfg.train = CN()
    cfg.train.tag = fyaml["train"]["tag"]
    cfg.train.batch_size = fyaml["train"]["batch_size"]
    cfg.train.epochs = fyaml["train"]["epochs"]
    cfg.train.lr = fyaml["train"]["lr"]
    cfg.train.lr_decay = fyaml["train"]["lr_decay"]
    cfg.train.weight_decay = fyaml["train"]["weight_decay"]
    cfg.train.alpha = fyaml["train"]["alpha"]
    cfg.train.beta = fyaml["train"]["beta"]
    cfg.train.gamma = fyaml["train"]["gamma"]

    return cfg

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def show_cfg(cfg, model, is_test=False):
    os.system('clear')
    print("*Model info")
    print(f" -Backbone: {cfg.model.backbone}")
    print(f" -Bin size: {cfg.model.bin_size}")
    print(f" -Decoder Type: {cfg.model.decoder_type}")
    print(f" -Decoder Size: {cfg.model.decoder_dim}")
    print(f" -Num head: {cfg.model.num_head}")
    print(f" -Embedding Dim: {cfg.model.d_model}")
    print(f" -Model Params: {count_parameters(model)/1000000:.2f}M")
    print(f" -Input Image: {cfg.data.image_size}")
    print(f" -LiDAR Points: {cfg.data.lidar_size}")
    print("=========================")
    if not is_test:
        print("*Train info")
        print(f" -Epochs: {cfg.train.epochs}")
        print(f" -Batch Size: {cfg.train.batch_size}")
        print(f" -Adam Optimzier")
        print(f"   -Learning Rate: {cfg.train.lr}")
        print(f"   -LR Decay: {cfg.train.lr_decay}")
        print(f"   -Weight Decay: {cfg.train.weight_decay}")
        print("=========================")