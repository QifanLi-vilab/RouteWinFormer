import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch

from basicsr.models.archs.RouteWinFormer_arch import RouteWinFormerLocal
import cv2
import utils
from natsort import natsorted
from glob import glob

import lpips

alex = lpips.LPIPS(net='alex').cuda()

parser = argparse.ArgumentParser(description='Single Image Defocus Deblurring using RouteWinFormer')

parser.add_argument('--input_dir', default=r'/mnt/Data/qfl/Datasets/DPDD/test/', type=str,
                    help='Directory of validation images')
parser.add_argument('--result_dir', default='./', type=str,
                    help='Directory for results')
parser.add_argument('--dim', default=32, type=int, help='Model Dimension')
parser.add_argument('--weights', default=r'../Pretrained/DPDD32.pth', type=str,
                    help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
args = parser.parse_args()

####### Load yaml #######
if args.dim == 32:
    yaml_file = '../options/test/DPDD/RouteWinFormer-width32.yml'
elif args.dim == 64:
    yaml_file = '../options/test/DPDD/RouteWinFormer-width64.yml'
else:
    assert "only import two size models, dim=32 or dim=64 !"
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################
model_restoration = RouteWinFormerLocal(**x['network_g'], factor_h=3.0, factor_w=3.0)

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

result_dir = os.path.join(args.result_dir, 'DPDD')
if args.save_images:
    os.makedirs(result_dir, exist_ok=True)

filesI = natsorted(glob(os.path.join(args.input_dir, 'input', '*.png')))
filesC = natsorted(glob(os.path.join(args.input_dir, 'target', '*.png')))

indoor_labels = np.load(r'/mnt/Data/qfl/Datasets/DPDD/test/indoor_labels.npy')
outdoor_labels = np.load(r'/mnt/Data/qfl/Datasets/DPDD/test/outdoor_labels.npy')

psnr, mae, ssim, pips = [], [], [], []
with torch.no_grad():
    for fileI, fileC in tqdm(zip(filesI, filesC), total=len(filesC)):

        imgI = np.float32(utils.load_img(fileI)) / 255.
        imgC = np.float32(utils.load_img(fileC)) / 255.

        patchI = torch.from_numpy(imgI).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        patchC = torch.from_numpy(imgC).unsqueeze(0).permute(0, 3, 1, 2).cuda()

        restored = model_restoration(patchI)[-1]
        restored = torch.clamp(restored, 0, 1)
        pips.append(alex(patchC, restored, normalize=True).item())

        restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        psnr.append(utils.PSNR(imgC, restored))
        mae.append(utils.MAE(imgC, restored))
        ssim.append(utils.SSIM(imgC, restored, ))
        if args.save_images:
            save_file = os.path.join(result_dir, os.path.split(fileC)[-1])
            restored = np.uint8((restored * 255).round())
            utils.save_img(save_file, restored)

psnr, mae, ssim, pips = np.array(psnr), np.array(mae), np.array(ssim), np.array(pips)

psnr_indoor, mae_indoor, ssim_indoor, pips_indoor = psnr[indoor_labels - 1], mae[indoor_labels - 1], ssim[
    indoor_labels - 1], pips[indoor_labels - 1]
psnr_outdoor, mae_outdoor, ssim_outdoor, pips_outdoor = psnr[outdoor_labels - 1], mae[outdoor_labels - 1], ssim[
    outdoor_labels - 1], pips[outdoor_labels - 1]
print("Overall: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr), np.mean(ssim), np.mean(mae),
                                                                    np.mean(pips)))
print("Indoor:  PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_indoor), np.mean(ssim_indoor),
                                                                    np.mean(mae_indoor), np.mean(pips_indoor)))
print("Outdoor: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_outdoor), np.mean(ssim_outdoor),
                                                                    np.mean(mae_outdoor), np.mean(pips_outdoor)))
