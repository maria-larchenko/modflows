import os
import numpy as np
import torch
import torch.nn as nn
import torchvision

from PIL import Image
from torchvision.transforms import v2
from torch.distributions.multivariate_normal import MultivariateNormal


class NeuralODE(nn.Module):
    
    def __init__(self, input_dim, device, hidden=32):
        super().__init__()
        self.device = device
        self.hidden = hidden
        self.input_dim = input_dim + 1
        self.output_dim = input_dim
        self.activation = nn.Tanh()
        self.layer_1 = nn.Linear(self.input_dim, self.hidden, bias=1)
        self.layer_2 = nn.Linear(self.hidden, self.output_dim, bias=1)
        self.shapes = [
          self.layer_1.weight.shape,
          self.layer_1.bias.shape,
          self.layer_2.weight.shape,
          self.layer_2.bias.shape,
         ]
        self.splits = [0,
          self.input_dim * hidden, 
          self.input_dim * hidden + hidden, 
          self.input_dim * hidden + hidden + self.output_dim * hidden,
          self.input_dim * hidden + hidden + self.output_dim * hidden + self.output_dim,
         ]
        self.total_params = sum(p.numel() for p in self.parameters())
        self.to(self.device)
        
    def set_weights(self, e):
        assert len(e) == self.total_params
        splits = self.splits
        shapes = self.shapes
        e0 = e[splits[0]:splits[1]].reshape(shapes[0])
        e1 = e[splits[1]:splits[2]].reshape(shapes[1])
        e2 = e[splits[2]:splits[3]].reshape(shapes[2])
        e3 = e[splits[3]:splits[4]].reshape(shapes[3])
        mask_dict = {
            'layer_1.weight': e0,
            'layer_1.bias': e1,
            'layer_2.weight': e2,
            'layer_2.bias': e3
        }
        self.load_state_dict(mask_dict)
        self.to(self.device)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        xt = self.layer_1(xt)
        xt = self.activation(xt)
        xt = self.layer_2(xt)
        return xt

    @torch.no_grad()
    def sample(self, x0, N=10_000, strength=1.0):
        sample_size = len(x0) 
        z = x0.detach().clone()
        dt = 1.0 / N
        for i in range(N):
            t = torch.ones((sample_size, 1)) * i / N
            t = t.to(self.device)
            z = z.to(self.device)
            pred = self.forward(z, t)
            #eps = 0.03
            #noise = torch.randn_like(pred)
            #z = z.detach().clone() + pred * dt + eps*noise*torch.sqrt(t*(1-t))
            z = z.detach().clone() + pred * dt
            if i > int(strength * N):
                break
        return z.detach().clone()

    @torch.no_grad()
    def inv_sample(self, x0, N=10_000, strength=1.0):
        sample_size = len(x0)
        z = x0.detach().clone()
        dt = 1.0 / N
        for i in range(N):
            t = torch.ones((sample_size, 1)) * i / N
            t = t.to(self.device)
            z = z.to(self.device)
            pred = self.forward(z, t)
            z = z.detach().clone() - pred * dt
            if i > int(strength * N):
                break
        return z.detach().clone()

def train_ode(model, lr, base_x, targ_x, samples, sample_size, tt=None, text=None, shuffle=True):
    data_size = base_x.shape[0]
    device = model.device
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for sn in range(samples):
        optim.zero_grad()

        indx = torch.randperm(data_size)[:sample_size]
        x_1 = targ_x[indx].to(device)
        
        if shuffle:
            indx = torch.randperm(data_size)[:sample_size]
        x_0 = base_x[indx].to(device)
        
        t = torch.rand((sample_size, 1), device=device)
        
        z_t = t * x_1 + (1.0 - t) * x_0
        v = x_1 - x_0
        v_pred = model(z_t, t)
        loss = torch.mean((v - v_pred)**2)
        
        loss.backward()
        optim.step()
        if tt is not None and sn % 100 == 0:
            tt.set_description(f"{text} {sn}/{samples} || loss: {loss.detach().cpu()}")
            tt.refresh()


def normal_to_uniform(x):
    return (torch.special.erf(x / np.sqrt(2)) + 1) / 2


def uniform_latent(dim, data_size):
    base_mu = torch.zeros(dim)
    base_cov = torch.eye(dim)
    latent_dist = MultivariateNormal(base_mu, base_cov)
    x = latent_dist.rsample(sample_shape=(data_size,))
    return normal_to_uniform(x)
    

def create_save_path(filepath, dataset_root, flows_root):
    """
    Takes on input filepath to an image in a dataset_root folder
    and creates the same dirs in a flows_root folder to save a model.
    """
    filename = filepath.split("/")[-1]
    start_char = len(dataset_root)
    last_char = len(filepath) - len(filename)
    savepath = flows_root + filepath[start_char:last_char]
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    return savepath



