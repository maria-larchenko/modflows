import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm, trange
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torch.distributions.multivariate_normal import MultivariateNormal
from mpl_toolkits.mplot3d import Axes3D

from einops import einsum
from src.neural_ode import NeuralODE, train_ode
from src.encoder import Encoder, enc_preprocess
from src.inference import print_images

from src.encoder import INPUT_SIZE


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATA = 'data/merged/'
FLOW = 'check_points/latent_flow_merged_8195/'
SAVE = ['check_points/V7_encoder/','_merged_8195_encoder']

pretrained_path = SAVE[0] + "2024.04.28 14-08-55_merged_8195_encoder_epoch_700000"
PRETRAINED_EPOCH = 700000


def get_encoder_path(epoch):
    timestamp = datetime.now().strftime('%Y.%m.%d %H-%M-%S')
    return SAVE[0] + timestamp + SAVE[1] + f"_epoch_{1 + epoch + PRETRAINED_EPOCH}"


def load_filenames(path):
    dataset_filenames = []
    for dirpath, dirnames, filenames in os.walk(path):
        print(f"{dirpath} : {len(filenames)}")
        for fname in filenames:
            dataset_filenames.append(dirpath + "/" + fname)
    print(f"total: {len(dataset_filenames)}")
    return dataset_filenames


def get_flow_id(filepath):
    filename = filepath.split("/")[-1].split(".")[0]
    return filename + "_model"


def get_flow_path(filepath, dataset_root, flows_root):
    """
    Takes on input filepath to an image in a dataset_root folder
    and returns the corresponded model in a flows_root.
    """
    filename = filepath.split("/")[-1]
    start_char = len(dataset_root)
    last_char = len(filepath) - len(filename)
    savepath = flows_root + filepath[start_char:last_char]
    flow_id = get_flow_id(filepath)    
    return savepath + flow_id

## ------------ hyper-parameters
lr = 1e-4
epochs = 200_000
batch = 8
sample_size = 512 * 20
steps = 100

# --------- load the model
trained_model = NeuralODE(input_dim=3, device=device, hidden=1024)
k_dim = trained_model.total_params

print(f"k_dim for encoder: {k_dim}")

encoder = Encoder(k_dim=k_dim, input_dim=4, hidden=1024, output_dim=3, device=device)
if pretrained_path:
    encoder.load_state_dict(torch.load(pretrained_path, map_location=device))
    print(f"FROM CHECKPOINT: {pretrained_path}")
enc_optim = torch.optim.Adam(encoder.parameters(), lr=lr)
data_size = INPUT_SIZE**2
encoder.train()

# ------------ load images
print(f"DATA: {DATA}")
print(f"FLOW: {FLOW}")
print(f"savepath format: {get_encoder_path(0)}")

dataset_filenames = load_filenames(DATA)
dataset_size = len(dataset_filenames)

model_paths = [get_flow_path(im_path, DATA, FLOW) for im_path in dataset_filenames]
models_params = [torch.load(flow_path, map_location=device) for flow_path in model_paths]

print(f"CHECK: \n -- img  {dataset_filenames[-5]} \n -- flow {model_paths[-5]}")

# --------- train the encoder
loss_track = [np.nan]
tt = trange(epochs, desc='>>> Next train step <<<', leave=True)
for epoch in tt:

    enc_optim.zero_grad()

    im_nums = torch.randperm(dataset_size)[:batch]

    param_list = [torch.load(model_paths[i], map_location=device) for i in im_nums]
    
    im_pil = [Image.open(dataset_filenames[i]).convert('RGB') for i in im_nums]
    im_enc = [enc_preprocess(im, rand_trans=True) for im in im_pil]
    im_flats = [im.reshape(INPUT_SIZE**2, 3) for im in im_enc]
    
    im_batch = torch.stack(im_enc, dim=0).to(device) # [32, 3, 256, 256]
    e = encoder(im_batch)  # [32, 8195]

    # generate targets
    t_list = []
    z_list = []
    v_list = []
    for im_base, trained_param in zip(im_flats, param_list):
        trained_model.load_state_dict(trained_param)
        trained_model.requires_grad_(False)
        
        im_base = im_base.to(device)
        indx = torch.randperm(data_size)[:sample_size]
        x_0 = im_base[indx].to(device)
        x_1 = trained_model.sample(x_0, N=steps)
        t = torch.rand((sample_size, 1), device=device)

        z_t = t * x_1 + (1.0 - t) * x_0
        v_trained = trained_model(z_t, t)

        t_list.append(t)
        z_list.append(z_t)
        v_list.append(v_trained)

    t_batch = torch.stack(t_list, dim=0).to(device)
    z_batch = torch.stack(z_list, dim=0).to(device)
    v_batch = torch.stack(v_list, dim=0).to(device)
    v_pred = encoder.apply_e(e, z_batch, t_batch)

    enc_loss = torch.mean((v_pred - v_batch)**2)

    loss_track.append(enc_loss.detach().cpu())
    tt.set_description(f"{epoch} || enc_loss: {loss_track[-1]}")
    tt.refresh()

    enc_loss.backward()
    enc_optim.step()

    if epoch > 0 and epoch % 3000 == 0:
        np.savetxt(f'{SAVE[0]}loss_track.txt', loss_track)
        torch.save(encoder.state_dict(), get_encoder_path(epoch))
        print(f'\n epoch: {epoch}  saved: {get_encoder_path(epoch)}')
        # print(f'\n epoch: {epoch}    ode models: {len(model_paths)}')

np.savetxt(f'{SAVE[0]}loss_track.txt', loss_track)
torch.save(encoder.state_dict(), get_encoder_path(epoch))
print("\n ENCODER: FINISHED")




