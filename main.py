import torch
import torch.nn as nn
import models
import dataloaders
import tools
from tqdm import tqdm
from flops_profiler.profiler import get_model_profile

class model_banch(nn.Module):
    def __init__(self, model):
        super(model_banch, self).__init__()
        self.model = model
    
    def forward(self, x):
        data = {"image":torch.rand([1,3,228,304]).to(cfg.device), 
                "lidar":torch.rand([1,1,cfg.data.lidar_size]).to(cfg.device)}
        out = self.model(data)

cfg = cfg = tools.get_attr("/root/MDE_LiDAR/config/baseline.yaml")

model = models.DepthNet(cfg).to(cfg.device)
# model = models.DepthNet_Nobins(cfg).to(cfg.device)

model.eval()

tools.show_cfg(cfg, model)

with torch.no_grad():
    model_b = model_banch(model)
    flops, macs, params = get_model_profile(model=model_b, input_shape=(1,1), print_profile=False)
    print(f"FLOPs: {flops/1000000000:.2f}G")
    print(f" MACs: {macs/1000000000:.2f}G")

test_loader = dataloaders.getTest_Data(batch_size=1, root_path="/root", lp=cfg.data.lidar_size)
train_loader = dataloaders.getTrain_Data(batch_size=cfg.train.batch_size, root_path="/root", lp=cfg.data.lidar_size)

for batch in tqdm(test_loader):
    tools.to_device(batch, cfg.device)

    out = model(batch)
