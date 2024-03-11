import torch
import models
import dataloaders
import tools
import os

# Get Trained File Path
args = tools.get_config()
root_path = args.root_path
log_file = args.config
ld = os.listdir(f"{root_path}/MDE_LiDAR/log/{log_file}")
config_yaml = [f for f in ld if f.endswith(".yaml")][0]
weight = [f for f in ld if f.endswith(".pth.tar")][0]

# Load Config File
cfg = tools.get_attr(f"{root_path}/MDE_LiDAR/log/{log_file}/{config_yaml}") # Load Test Model

# Model Load & Initialization
model = models.DepthNet(cfg).to(device=cfg.device)
model.load_state_dict(torch.load(f"{root_path}/MDE_LiDAR/log/{log_file}/{weight}"))
model.eval()

# Dataload
test_loader = dataloaders.getTest_Data(batch_size=1, root_path=root_path, lp=cfg.data.lidar_size)

tools.show_cfg(cfg, model, True)

# Loss Funtions
Loss = models.Losses(cfg).to(device=cfg.device)

if __name__ == "__main__":
    for i, b in enumerate(test_loader):
        tools.to_device(b, cfg.device)
        o, c = model(b)
        break
    path = f"{root_path}/output/{config_yaml[:-5]}-{cfg.train.tag}-eval"
    tools.visualization(test_loader, model, path, cfg.device)