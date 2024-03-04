import torchvision
from .model import DepthNet, DepthNet_Nobins
from .cross_attention import Cross_Attention_Block
from .decoder import Decoder, Decoder_Inter
from .loss import *

__model_info__ = {
    "mobilenetv2" : [[1, 3, 6, 13, 17], [320, 96, 32, 24, 16]],
    "mobilenetv3_small" : [[0, 1, 3, 6, 11], [96, 40, 24, 16, 16]],
    "mobilenetv3_large" : [[1, 3, 6, 12, 15], [160, 112, 40, 24, 16]],
    "efficientnetb0" : [[1, 2, 3, 5, 7], [320, 112, 40, 24, 16]],
}

def get_model(backbone):
    if backbone == "mobilenetv2":
        return torchvision.models.mobilenet_v2(weights="DEFAULT"), __model_info__[backbone]
    elif backbone == "efficientnetb0":
        return torchvision.models.efficientnet_b0(weights="DEFAULT"), __model_info__[backbone]
    elif backbone == "mobilenetv3_small":
        return torchvision.models.mobilenet_v3_small(weights="DEFAULT"), __model_info__[backbone]
    elif backbone == "mobilenetv3_large":
        return torchvision.models.mobilenet_v3_large(weights="DEFAULT"), __model_info__[backbone]
    else:
        raise Exception(TypeError, "Undefined Backdone Network")

def sel_decoder(cfg, info):
    if cfg.model.decoder_type == 0:
        return Decoder(cfg, info)
    else:
        return Decoder_Inter(cfg, info)