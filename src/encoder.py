import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.transforms import v2
from einops import einsum


INPUT_SIZE = 256
# INPUT_SIZE = 528


def enc_preprocess(pil_image, crop=False, image=False, rand_trans=False):
    im = pil_image
    im_size = (INPUT_SIZE, INPUT_SIZE)
    enc_shape = (3, INPUT_SIZE, INPUT_SIZE)  #  (1, 3, INPUT_SIZE, INPUT_SIZE)
    # v2 currently accepts only PIL Images
    if crop:
        crop = min(im.size)
        im = v2.CenterCrop(crop)(im)
    im = v2.Resize(im_size)(im)
    if rand_trans:
        im = v2.RandomHorizontalFlip(p=0.5)(im)
        im = v2.functional.rotate(im, angle=np.random.choice([0, 90, -90, 180]))
    if image:
        return im
    im = np.array(im, dtype=np.float32) / 255
    im = im.reshape(enc_shape)
    return torch.tensor(im)


class Encoder(nn.Module):
    
    def __init__(self, k_dim, input_dim, hidden, output_dim, device):
        super().__init__()
        self.k_dim = k_dim
        self.model = torchvision.models.efficientnet_b0(num_classes=k_dim)
        self.resize = torchvision.models.efficientnet.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()  # INPUT_SIZE = 256
        
        # self.model = torchvision.models.efficientnet_b6(num_classes=k_dim)
        # self.resize = torchvision.models.efficientnet.EfficientNet_B6_Weights.IMAGENET1K_V1.transforms()  # INPUT_SIZE = 528
        
        self.device = device
#       # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, 
#       # self.conv = nn.Conv2d(6, 3, (3, 3), stride=1, padding=1)
        
        self.input_dim = input_dim 
        self.hidden = hidden
        self.output_dim = output_dim
        self.splits = [0,
          input_dim * hidden, 
          input_dim * hidden + hidden, 
          input_dim * hidden + hidden + output_dim * hidden,
          input_dim * hidden + hidden + output_dim * hidden + output_dim,
         ]
        self.to(device)
        
    def forward(self, im1):
        # the input should be [batch_size, channels, height, width]
        with torch.no_grad():
            im1 = self.resize(im1)
        return self.model(im1)
    
    def apply_e(self, e, x, t):
        splits = self.splits
        batch_size = e.shape[0]
        shapes = [
          torch.Size([batch_size, self.hidden, self.input_dim]),
          torch.Size([batch_size, self.hidden]),
          torch.Size([batch_size, self.output_dim, self.hidden]),
          torch.Size([batch_size, self.output_dim]),
         ]
        # e = [batch_size, params_size]
        e0 = e[:, splits[0]:splits[1]].reshape(shapes[0])
        e1 = e[:, splits[1]:splits[2]].reshape(shapes[1])
        e2 = e[:, splits[2]:splits[3]].reshape(shapes[2])
        e3 = e[:, splits[3]:splits[4]].reshape(shapes[3])
        e1 = e1.unsqueeze(1)
        e3 = e3.unsqueeze(1)
        # x = [batch_size, sample_size, channels]
        # t = [batch_size, sample_size, 1]
        xt = torch.cat([x, t], dim=-1)
        xt = einsum(xt, e0, 'i j k, i n k -> i j n') + e1
        xt = torch.tanh(xt)
        xt = einsum(xt, e2, 'i j k, i n k -> i j n') + e3
        return xt
    
    
    
    