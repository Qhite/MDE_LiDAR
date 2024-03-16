import torch.nn as nn
import torch.nn.functional as F
import models

class DepthNet(nn.Module):
    def __init__(self, cfg):
        super(DepthNet, self).__init__()
        self.bin_size = cfg.model.bin_size

        # Build Model
        self.model, self.info = models.get_model(cfg.model.backbone)
        self.encoder = self.model.get_submodule("features")
        self.binning = models.Cross_Attention_Block(cfg, self.info[1])
        self.decoder = models.sel_decoder(cfg, self.info[1])

        # Decoder Output Conv
        self.Conv_out = nn.Sequential(
            nn.Conv2d(cfg.model.decoder_dim, self.bin_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(),
        )
    
    def get_features(self, x):
        features = []
        for name, module in self.encoder.named_children():
            x = module(x)
            if int(name) in self.info[0]:
                features.append(x)
                if int(name) == self.info[0][-1]:
                    break
        return x, features

    def forward(self, data):
        x, y = data["image"], data["lidar"]

        _, f = self.get_features(x)
        
        out = self.decoder(f)
        out = self.Conv_out(out)
        out = F.softmax(out, dim=1) # Depth bin-probability map

        bins = self.binning({"feature":f[0], "lidar":y}) + 1e-3
        bins = bins / bins.sum(axis=1, keepdim=True)
        
        # Bin Centers
        bin_width = F.pad(bins, (1,0), mode="constant", value=1e-3)
        bin_edge = bin_width.cumsum(dim=1)
        centers = 0.5 * (bin_edge[:, :-1]+bin_edge[:, 1:])
        centers = centers.unsqueeze(2).unsqueeze(2)

        predict = (out * centers).sum(axis=1, keepdim=True)

        return predict, centers.squeeze()

class DepthNet_Nobins(nn.Module):
    def __init__(self, cfg):
        super(DepthNet_Nobins, self).__init__()
        self.bin_size = cfg.model.bin_size

        # Build Model
        self.model, self.info = models.get_model(cfg.model.backbone)
        self.encoder = self.model.get_submodule("features")
        self.decoder = models.sel_decoder(cfg, self.info[1])

        # Decoder Output Conv
        self.Conv_out = nn.Sequential(
            nn.Conv2d(cfg.model.decoder_dim, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )
    
    def get_features(self, x):
        features = []
        for name, module in self.encoder.named_children():
            x = module(x)
            if int(name) in self.info[0]:
                features.append(x)
                if int(name) == self.info[0][-1]:
                    break
        return x, features

    def forward(self, data):
        x, y = data["image"], data["lidar"]

        _, f = self.get_features(x)
        
        out = self.decoder(f)
        out = self.Conv_out(out)

        return out