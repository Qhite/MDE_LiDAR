import torch
import models
import dataloaders
import tools
from tqdm import tqdm
import pandas as pd

torch.manual_seed(13)
torch.cuda.manual_seed(13)

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic=True

# Get Config File Path
args = tools.get_config()
root_path = args.root_path
config_yaml = args.config

# Load Config File
cfg = tools.get_attr(f"{root_path}/MDE/config/{config_yaml}") # Load Train config

# Model Load & Initialization
model = models.DepthNet(cfg).to(device=cfg.device)

# Dataload
train_loader = dataloaders.getTrain_Data(batch_size=cfg.train.batch_size, root_path=root_path, lp=cfg.data.lidar_size)
test_loader = dataloaders.getTest_Data(batch_size=1, root_path=root_path, lp=cfg.data.lidar_size)

# Optimizer
enc = list(map(id, model.encoder.parameters()))
params = filter(lambda p: id(p) not in enc, model.parameters())

optimizer = torch.optim.Adam([{"params": model.encoder.parameters(), "lr": cfg.train.lr * 0.1},
                             {"params": params}],
                             lr=cfg.train.lr,
                             weight_decay=cfg.train.weight_decay
                             )

# Loss Funtions
SILL = models.SILogLoss().to(device=cfg.device)
BCL = models.BinsChamferLoss().to(device=cfg.device)
MML = models.MinMaxLoss().to(device=cfg.device)

# Show Configs
tools.show_cfg(cfg, model)

Epochs = cfg.train.epochs
logging = pd.DataFrame(columns=["Epoch", "loss(train)", "loss(val)", "delta1", "delta2", "delta3", "RMS", "Log", "Rel", "SqRel"])

# Train
def train():
    for epoch in range(Epochs):
        buff_l = torch.zeros(1)
        log_epoch = [epoch+1, 0] # For train logging

        model.train()
        tools.update_lr(optimizer, epoch, cfg, [5, 10, 15])

        train_tqdm = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, sample in train_tqdm:
            tools.to_device(sample, cfg.device)

            optimizer.zero_grad()

            output, centers = model(sample)
            target = sample["depth"]

            SIL_loss = SILL(output, target)
            BC_loss = BCL(centers, target)
            MM_loss = MML(centers, target)

            loss = cfg.train.alpha * SIL_loss + cfg.train.beta * BC_loss + cfg.train.gamma * MM_loss 
            loss.backward()
            
            optimizer.step()

            buff_l += loss.clone().detach().cpu()

            train_tqdm.set_description(f"Epoch {epoch+1:2d}/{Epochs:2d} | Loss {float(buff_l/i):.3f}")

        model.eval()
        val_logging = validate()

        log_epoch[1] = float(buff_l/len(train_loader))
        log_epoch.extend(val_logging)
        logging.loc[epoch] = log_epoch

    tools.save_model(args, model, logging)

def validate():
    with torch.no_grad():
        val_tqdm = tqdm(enumerate(test_loader), total=len(test_loader))
        buff_m = torch.zeros(7)
        buff_l = torch.zeros(1)

        for i, batch in val_tqdm:

            tools.to_device(batch, cfg.device)

            predict, centers = model(batch)
            t = batch["depth"].detach()
            p, c = predict.detach(), centers.detach()

            metrics = tools.cal_metric(p * 10, t * 10) # Set Depth unit to meter
            buff_m += metrics

            loss = cfg.train.alpha * SILL(p, t).detach() + cfg.train.beta * BCL(c, t).detach() + cfg.train.gamma * MML(c.unsqueeze(0), t).detach()
            buff_l += loss.cpu()

            val_tqdm.set_description(f"Delta_1 {float(buff_m[0]/i):.3f} | RMS {float(buff_m[3]/i):.3f} | REL {float(buff_m[5]/i):.3f} | loss {float(buff_l/i):.3f}")
        
        avg_loss = (buff_l/len(test_loader)).tolist()
        avg_errors = (buff_m/len(test_loader)).tolist()
        avg_loss.extend(avg_errors)
        
        tools.show_metric(avg_errors)
    print("==" * 50)

    return avg_loss

def visualization(data_loader):
    from torchvision.transforms.functional import to_pil_image
    import matplotlib.pyplot as plt
    import os
    import torch.nn.functional as F

    path = f"{root_path}/output/{config_yaml[:-5]}-{cfg.train.tag}"
    os.system(f"mkdir -p {path}")

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        tools.to_device(batch, cfg.device)
        predict, centers = model(batch)
        for j, (p, d) in enumerate(zip(predict, batch["depth"])):
            p = F.interpolate(p.unsqueeze(0), size=[228, 304], mode="nearest")
            out = to_pil_image(p.squeeze())
            plt.imsave(f"{path}/{i}_{j}_pr.png", out, cmap="magma")

            d = F.interpolate(d.unsqueeze(0), size=[228, 304], mode="nearest")
            out = to_pil_image(d.squeeze())
            plt.imsave(f"{path}/{i}_{j}_gt.png", out, cmap="magma")

            out = to_pil_image((p-d).abs().squeeze())
            plt.imsave(f"{path}/{i}_{j}_er.png", out, cmap="bwr")


if __name__ == "__main__":
    train()

    # Visualization
    visualization()