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
cfg = tools.get_attr(f"{root_path}/MDE_LiDAR/config/{config_yaml}") # Load Train config

# Model Load & Initialization
model = models.DepthNet(cfg).to(device=cfg.device)

# Dataload
train_loader = dataloaders.getTrain_Data(batch_size=cfg.train.batch_size, root_path=root_path, lp=cfg.data.lidar_size[-1], channel=cfg.data.lidar_size[0])
test_loader = dataloaders.getTest_Data(batch_size=1, root_path=root_path, lp=cfg.data.lidar_size[-1], channel=cfg.data.lidar_size[0])

# Optimizer
enc = list(map(id, model.encoder.parameters()))
params = filter(lambda p: id(p) not in enc, model.parameters())

optimizer = torch.optim.Adam([{"params": model.encoder.parameters(), "lr": cfg.train.lr * 0.1},
                             {"params": params}],
                             lr=cfg.train.lr,
                             weight_decay=cfg.train.weight_decay
                             )

# Loss Funtions
Loss = models.Losses(cfg).to(device=cfg.device)

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
        tools.update_lr(optimizer, epoch, cfg)

        train_tqdm = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, sample in train_tqdm:
            tools.to_device(sample, cfg.device)

            optimizer.zero_grad()

            output, centers = model(sample)

            loss = Loss(epoch, output, centers, sample)
            loss.backward()
            
            optimizer.step()

            buff_l += loss.clone().detach().cpu()

            train_tqdm.set_description(f"Epoch {epoch+1:2d}/{Epochs:2d} | Loss {float(buff_l/i):.3f}")

        model.eval()
        val_logging = validate(epoch)

        log_epoch[1] = float(buff_l/len(train_loader))
        log_epoch.extend(val_logging)
        logging.loc[epoch] = log_epoch

    tools.save_model(args, model, logging)

@torch.no_grad()
def validate(epoch):
    val_tqdm = tqdm(enumerate(test_loader), total=len(test_loader))
    buff_m = torch.zeros(7)
    buff_l = torch.zeros(1)

    for i, batch in val_tqdm:
        tools.to_device(batch, cfg.device)

        predict, centers = model(batch)
        t = batch["depth"]
        p, c = predict, centers

        metrics = tools.cal_metric(p * 10, t * 10) # Set Depth unit to meter
        buff_m += metrics

        loss = Loss(epoch, p, c.unsqueeze(0), batch)
        buff_l += loss.cpu()

        val_tqdm.set_description(f"Delta_1 {float(buff_m[0]/i):.3f} | RMS {float(buff_m[3]/i):.3f} | REL {float(buff_m[5]/i):.3f} | loss {float(buff_l/i):.3f}")
        # if i == len(test_loader)-1:
        #     print(c)
    
    avg_loss = (buff_l/len(test_loader)).tolist()
    avg_errors = (buff_m/len(test_loader)).tolist()
    avg_loss.extend(avg_errors)
    
    tools.show_metric(avg_errors)
    print("==" * 50)

    return avg_loss


if __name__ == "__main__":
    train()

    # Visualization
    # path = f"{root_path}/output/{config_yaml[:-5]}-{cfg.train.tag}"
    # tools.visualization(test_loader, model, path, cfg.device)