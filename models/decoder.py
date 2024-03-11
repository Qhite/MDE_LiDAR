import torch.nn as nn
import torch.nn.functional as F
import math

def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding, bias=False, groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.LeakyReLU(inplace=True),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.LeakyReLU(inplace=True), 
        )

class Decoder(nn.Module):
    def __init__(self, cfg=None, channels=[]):
        super(Decoder, self).__init__()
        self.H, self.W = cfg.data.image_size
        self.dim = cfg.model.decoder_dim
        
        self.conv0 = nn.Sequential(
            depthwise(channels[0], 3),
            pointwise(channels[0], self.dim*5)
        )

        self.Trans_conv1 = nn.ConvTranspose2d(self.dim*5, self.dim*4, kernel_size=5, stride=2, padding=2)
        self.skip_conv1 = nn.Sequential(
            depthwise(channels[1], 3),
            pointwise(channels[1], self.dim*4),
            nn.Dropout(0.1)
        )
        self.res_conv1 = nn.Sequential(
            depthwise(self.dim*4, 5),
            pointwise(self.dim*4, self.dim*4),
        )

        self.Trans_conv2 = nn.ConvTranspose2d(self.dim*4, self.dim*3, kernel_size=5, stride=2, padding=2)
        self.skip_conv2 = nn.Sequential(
            depthwise(channels[2], 3),
            pointwise(channels[2], self.dim*3),
            nn.Dropout(0.1)
        )
        self.res_conv2 = nn.Sequential(
            depthwise(self.dim*3, 5),
            pointwise(self.dim*3, self.dim*3)
        )

        self.Trans_conv3 = nn.ConvTranspose2d(self.dim*3, self.dim*2, kernel_size=5, stride=2, padding=2)
        self.skip_conv3 = nn.Sequential(
            depthwise(channels[3], 3),
            pointwise(channels[3], self.dim*2),
            nn.Dropout(0.1)
        )
        self.res_conv3 = nn.Sequential(
            depthwise(self.dim*2, 5),
            pointwise(self.dim*2, self.dim*2),
        )

        self.Trans_conv4 = nn.ConvTranspose2d(self.dim*2, self.dim, kernel_size=4, stride=2, padding=1)
        self.skip_conv4 = nn.Sequential(
            depthwise(channels[4], 3),
            pointwise(channels[4], self.dim),
            nn.Dropout(0.1)
        )
        self.res_conv4 = nn.Sequential(
            depthwise(self.dim, 5),
            pointwise(self.dim, self.dim),
        )

    def forward(self, f):
        x = self.conv0(f[-1]) # Decoder input

        x = self.Trans_conv1(x) + self.skip_conv1(f[-2])
        x = self.res_conv1(x)

        x = F.interpolate(self.Trans_conv2(x), size=[math.ceil(self.H/8), math.ceil(self.W/8)], mode="nearest") + self.skip_conv2(f[-3])
        x = self.res_conv2(x)

        x = F.interpolate(self.Trans_conv3(x), size=[math.ceil(self.H/4), math.ceil(self.W/4)], mode="nearest") + self.skip_conv3(f[-4])
        x = self.res_conv3(x)

        x = self.Trans_conv4(x) + self.skip_conv4(f[-5])
        x = self.res_conv4(x)

        return x
    
class Decoder_Inter(nn.Module): # Interpolation & Conv
    def __init__(self, cfg=None, channels=[]):
        super(Decoder_Inter, self).__init__()
        self.H, self.W = cfg.data.image_size
        self.dim = cfg.model.decoder_dim


        self.conv0 = nn.Sequential(
            depthwise(channels[0], 3),
            pointwise(channels[0], self.dim*5)
        )

        self.skip_conv1 = nn.Sequential(
            depthwise(channels[1], 3),
            pointwise(channels[1], self.dim*5),
            nn.Dropout(0.1)
        )
        self.res_conv1 = nn.Sequential(
            depthwise(self.dim*5, 5),
            pointwise(self.dim*5, self.dim*4),
        )

        self.skip_conv2 = nn.Sequential(
            depthwise(channels[2], 3),
            pointwise(channels[2], self.dim*4),
            nn.Dropout(0.1)
        )
        self.res_conv2 = nn.Sequential(
            depthwise(self.dim*4, 5),
            pointwise(self.dim*4, self.dim*3)
        )

        self.skip_conv3 = nn.Sequential(
            depthwise(channels[3], 3),
            pointwise(channels[3], self.dim*3),
            nn.Dropout(0.1)
        )
        self.res_conv3 = nn.Sequential(
            depthwise(self.dim*3, 5),
            pointwise(self.dim*3, self.dim*2),
        )

        self.skip_conv4 = nn.Sequential(
            depthwise(channels[4], 3),
            pointwise(channels[4], self.dim*2),
            nn.Dropout(0.1)
        )
        self.res_conv4 = nn.Sequential(
            depthwise(self.dim*2, 5),
            pointwise(self.dim*2, self.dim),
        )

        self.conv_out = nn.Sequential(
            depthwise(self.dim*2, 5),
            pointwise(self.dim*2, self.dim),
        )

    def forward(self, f):
        x = self.conv0(f[-1]) # Decoder input

        x = F.interpolate(x, size=[math.ceil(self.H/16), math.ceil(self.W/16)], mode='bilinear', align_corners=True) + self.skip_conv1(f[-2])
        x = self.res_conv1(x)

        x = F.interpolate(x, size=[math.ceil(self.H/8), math.ceil(self.W/8)], mode='bilinear', align_corners=True) + self.skip_conv2(f[-3])
        x = self.res_conv2(x)

        x = F.interpolate(x, size=[math.ceil(self.H/4), math.ceil(self.W/4)], mode='bilinear', align_corners=True) + self.skip_conv3(f[-4])
        x = self.res_conv3(x)

        x = F.interpolate(x, size=[math.ceil(self.H/2), math.ceil(self.W/2)], mode='bilinear', align_corners=True) + self.skip_conv4(f[-5])
        x = self.res_conv4(x)

        return x
