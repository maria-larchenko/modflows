import torch
import os
import shutil
import argparse

from tqdm import tqdm
from PIL import Image
from src.encoder import Encoder
from src.inference import run_inference, run_inference_flow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, default="./data/content/", help='Content folder')
    parser.add_argument("--style", type=str, default="./data/style/", help='Reference folder')
    parser.add_argument("--output", type=str, default="./data/output/", help='Output folder')
    parser.add_argument("--device_name", type=str, default="cuda:0", help='Device, cpu or cuda:{number}') 
    parser.add_argument("--strength", type=float, default=1.0, help='Strength of effected, from 0 to 1')
    parser.add_argument("--steps", type=int, default=8, help='Number of flow steps, from 2 to 100')
    
    args = parser.parse_args()
    content_dir = args.content
    style_dir = args.style
    results_dir = args.output

    strength = args.strength
    steps = args.steps
    device_name = args.device_name

    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

    print(f"Inference run on device: {device}")
    
    encoder = Encoder(k_dim=8195, input_dim=4, hidden=1024, output_dim=3, device=device)
    enc_path = './checkpoints/2024.04.28 14-08-55_merged_8195_encoder_epoch_700000.pt'
    enc_params = torch.load(enc_path, map_location=device)
    encoder.load_state_dict(enc_params)
    
    if os.path.exists(content_dir + '.ipynb_checkpoints'):
        shutil.rmtree(content_dir + '.ipynb_checkpoints')
    if os.path.exists(style_dir + '.ipynb_checkpoints'):
        shutil.rmtree(style_dir + '.ipynb_checkpoints')
        
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    content_names = sorted(os.listdir(content_dir))
    style_names = sorted(os.listdir(style_dir))

    assert len(content_names) == len(style_names), f'{len(content_names)} {len(style_names)}'
    assert 0 <= strength <= 1.0
    assert 2 <= steps <= 100
    
    print(f"steps: {steps}")
    if strength != 1.0:
        print(f"strength: {strength}")
        print(f"actual steps: {int(strength * steps)}")
    
    for i in tqdm(range(len(content_names))):
        cont_name = content_names[i]
        style_name = style_names[i]
        # print(f"{i}/{N}:: ", cont_name, "   --------- to ---------  ", style_name)
        im_1 = content_dir + cont_name
        im_2 = style_dir + style_name
        imgs = run_inference(encoder, device, im_1, im_2, enc_steps=steps, strength=strength);
        im = imgs[2].save(results_dir + cont_name)
    print("DONE")
