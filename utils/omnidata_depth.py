# adapted from https://github.com/EPFL-VILAB/omnidata
import torch
import torch.nn.functional as F
import torch.nn as nn 
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys

from omnidata_tools.torch.modules.unet import UNet
from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel

class DepthEstimator(nn.Module):
    def __init__(self, device='cuda'):
        pretrained_path = '/home/yujie/yujie_codes/defocus_gaussian/omnidata_tools/pretained_models/omnidata_dpt_depth_v2.ckpt'
        super(DepthEstimator, self).__init__()
        self.model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
        self.device = device 
        self.load_ckpt(pretrained_path)
        print('we use omnidata depth model')
    
    def load_ckpt(self, weights_path='/home/yujie/yujie_codes/defocus_gaussian/omnidata_tools/pretained_models/'):
        map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
        checkpoint = torch.load(weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
    
    def img_trans(self, input_img, img_size=384):
        resized_img = torch.clamp(torch.nn.functional.interpolate(
        input_img,
        size=(img_size, img_size),
        mode="bicubic",
        align_corners=False), 0.0, 1.0)
        return resized_img

    def inference(self, input_img):
        if len(input_img.shape) == 3:
            input_img = input_img.unsqueeze(0)
        if input_img.shape[-1] == 3 or input_img.shape[-1] == 1:
            input_img = input_img.permute(0, 3, 1, 2)
        if input_img.shape[1] == 1:
            input_img = input_img.repeat_interleave(3,1)
        h, w = input_img.shape[2:4]
        if h <= 512 and w <= 512:
            resized_img = self.img_trans(input_img)
        else:
            resized_img = self.img_trans(input_img, img_size=512)
        # predict the depth
        #print('shape of resized image', resized_img.shape)
        output = self.model(resized_img).clamp(min=0, max=1)
        #print('output.shape', output.shape)
        # resize the prediction back to the original resolution
        output = torch.nn.functional.interpolate(output.unsqueeze(0), size=(h, w), mode='bicubic', align_corners=False)
        output = output.squeeze(0)
        return output

    def forward(self, input_img, mode='test'):
        if mode == 'test':
            with torch.no_grad():
                return self.inference(input_img)
        elif mode == 'train':
            return self.inference(input_img)
            
        

'''
elif args.task == 'depth':
    image_size = 384
    pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
    # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
    model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=0.5, std=0.5)])

else:
    print("task should be one of the following: normal, depth")
    sys.exit()

trans_rgb = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                transforms.CenterCrop(image_size),
                                ])


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def save_outputs(img_path, output_file_name):
    with torch.no_grad():
        save_path = os.path.join(args.output_path, f'{output_file_name}_{args.task}.png')

        print(f'Reading input {img_path} ...')
        img = Image.open(img_path)

        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

        rgb_path = os.path.join(args.output_path, f'{output_file_name}_rgb.png')
        trans_rgb(img).save(rgb_path)

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3,1)

        output = model(img_tensor).clamp(min=0, max=1)

        if args.task == 'depth':
            #output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
            output = output.clamp(0,1)
            
            np.save(save_path.replace('.png', '.npy'), output.detach().cpu().numpy()[0])
            
            #output = 1 - output
#             output = standardize_depth_map(output)
            plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')
            
        else:
            #import pdb; pdb.set_trace()
            np.save(save_path.replace('.png', '.npy'), output.detach().cpu().numpy()[0])
            trans_topil(output[0]).save(save_path)
            
        print(f'Writing output {save_path} ...')


img_path = Path(args.img_path)
if img_path.is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif img_path.is_dir():
    for f in glob.glob(args.img_path+'/*'):
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
else:
    print("invalid file path!")
    sys.exit()
'''