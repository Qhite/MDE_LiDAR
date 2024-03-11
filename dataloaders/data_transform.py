import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import random

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class Resize():
    def __init__(self, size=(320, 240)):
        self.size = size
    
    def __call__(self, data):
        image, depth, lidar = data["image"], data["depth"], data['lidar']
        return {"image": self.resizing(image), "depth": self.resizing(depth), "lidar": lidar}
    
    def resizing(self, img):
        if not isinstance(self.size, (tuple, list)):
            raise TypeError("Size must be Tuple or list")
        
        if not _is_pil_image(img):
            raise TypeError("Input image should be PIL type")
        
        return img.resize(self.size)
    
class RandomHorizontalFlip(object):
    def __call__(self, data):
        image, depth, lidar = data['image'], data['depth'], data['lidar']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

            if data['lidar'] != None:
                lidar = data['lidar'][::-1]

        return {'image': image, 'depth': depth, 'lidar': lidar}
    
class CenterCrop(object):
    def __init__(self, size_image=[304, 228], size_depth=[304, 228]):
        self.size_image = size_image
        self.size_depth = size_depth

    def __call__(self, data):
        image, depth, lidar = data['image'], data['depth'], data['lidar']

        image = self.centerCrop(image, self.size_image)
        depth = self.centerCrop(depth, self.size_image)

        dw, dh = self.size_depth
        depth = depth.resize((dw, dh))

        return {'image': image, 'depth': depth, 'lidar': lidar}

    def centerCrop(self, image, size):
        w, h = image.size

        cw, ch = size

        if w == cw and h == ch:
            return image
        
        x_start = int(round((w - cw) / 2.))
        y_start = int(round((h - ch) / 2.))

        image = image.crop((x_start, y_start, cw + x_start, ch + y_start))

        return image

class ToTensor():
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, data):
        image, depth = data["image"], data["depth"]

        image = transforms.ToTensor()(image).float()

        if self.is_test:
            depth = transforms.ToTensor()(depth).float() / 10000.
        else:
            depth = transforms.ToTensor()(depth).float()
        
        if data["lidar"] != None:
            lidar = torch.tensor(data["lidar"]).float().unsqueeze(0) / 10000.
        else:
            lidar = data["lidar"]
        
        return {"image": image, "depth": depth, "lidar": lidar}

class Lighting(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, data):
        image, depth, lidar = data['image'], data['depth'], data['lidar']

        if self.alphastd == 0:
            return image

        alpha = image.new(3).normal_(0, self.alphastd)

        rgb = self.eigvec.type_as(image).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        image = image.add(rgb.view(3, 1, 1).expand_as(image))

        return {'image': image, 'depth': depth, 'lidar': lidar}

class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587*gs[1]).add_(0.114*gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs

class Saturation(object):
    def __init__(self, val):
        self.val = val

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.val, self.val)
        return img.lerp(gs, alpha)

class Brightness(object):
    def __init__(self, val):
        self.val = val

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.val, self.val)

        return img.lerp(gs, alpha)

class Contrast(object):
    def __init__(self, val):
        self.val = val

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.val, self.val)
        return img.lerp(gs, alpha)

class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))
    
    def __call__(self, data):
        image, depth, lidar = data['image'], data['depth'], data['lidar']

        if self.transforms is None:
            return {'image': image, 'depth': depth, 'lidar': lidar}

        for i in torch.randperm(len(self.transforms)):
            image = self.transforms[i](image)

        return {'image': image, 'depth': depth, 'lidar': lidar}

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        image, depth, lidar = data['image'], data['depth'], data['lidar']

        image = transforms.Normalize(self.mean, self.std)(image)

        return {'image': image, 'depth': depth, 'lidar': lidar}

class Add_LiDAR(object):
    def __init__(self, lp, offset=0, noise=None):
        self.lp = lp
        self.offset = offset
        self.noise = noise
        pass
    
    def __call__(self, data):
        image, depth = data['image'], data['depth']
        mid_idx = depth.size(2)//2 + self.offset

        if self.noise != None:
            masked = depth[0, mid_idx-self.noise:mid_idx+self.noise+1, :].transpose(1, 0).contiguous()
            mid = []
            for m in masked:
                mid.append( m[np.random.randint(0, self.noise*2+1)] )
            mid = torch.tensor(mid).squeeze()
        else:
            mid = depth[:, mid_idx, :].squeeze()

        if depth.size(-1) > self.lp:
            stride = mid.size(0) // self.lp
            pad = mid.size(0) % self.lp
            mid = mid[pad//2 : -pad//2 : stride].unsqueeze(0)
        else:
            raise("Too many LiDAR Points!")
        
        return {'image': image, 'depth': depth, 'lidar': mid}