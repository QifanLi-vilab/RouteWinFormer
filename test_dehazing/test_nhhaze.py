import os
import argparse
from glob import glob
from tqdm import tqdm
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from natsort import natsorted
from skimage import img_as_ubyte
from basicsr.models.archs.RouteWinFormer_arch import RouteWinFormer
from pytorch_msssim import ssim
from utils import save_img
import utils

def parse_args():
    parser = argparse.ArgumentParser(description='Image Dehazing using RouteWinFormer')
    parser.add_argument('--input_dir', default=r'/mnt/Data/qfl/Datasets/Dehazing/', type=str,
                        help='Directory of validation images')
    parser.add_argument('--result_dir', default='./', type=str, help='Directory for results')
    parser.add_argument('--weights',
                        default=r'../Pretrained/NHHAZE.pth',
                        type=str, help='Path to weights')
    parser.add_argument('--dataset', default='NH-HAZE', type=str,
                        help='Test Dataset')
    parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
    return parser.parse_args()


class ImageDataset(Dataset):
    def __init__(self, input_files, target_files):
        self.input_files = input_files
        self.target_files = target_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        imgI = np.float32(utils.load_img(self.input_files[idx])) / 255.
        imgT = np.float32(utils.load_img(self.target_files[idx])) / 255.
        imgI = torch.from_numpy(imgI).permute(2, 0, 1)
        imgT = torch.from_numpy(imgT).permute(2, 0, 1)
        filename = os.path.splitext(os.path.split(self.target_files[idx])[-1])[0] + '.png'
        return imgI, imgT, filename


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'NH-HAZE':
        yaml_file = '../options/test/DeHazing/RouteWinFormer-NHHAZE-width48.yml'
    else:
        assert "dataset name error!"

    with open(yaml_file, 'r') as f:
        x = yaml.load(f, Loader=yaml.Loader)

    s = x['network_g'].pop('type')

    model_restoration = RouteWinFormer(**x['network_g'])
    checkpoint = torch.load(args.weights, map_location='cpu')
    model_restoration.load_state_dict(checkpoint['params'])
    print("===>Testing using weights: ", args.weights)

    model_restoration = model_restoration.to(device)
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    dataset = args.dataset
    result_dir = os.path.join(args.result_dir, dataset)

    inp_dir = os.path.join(args.input_dir, dataset, 'test')
    filesI = natsorted(glob(os.path.join(inp_dir, 'input', '*.png')) + glob(os.path.join(inp_dir, 'input', '*.jpg')))
    filesT = natsorted(glob(os.path.join(inp_dir, 'target', '*.png')) + glob(os.path.join(inp_dir, 'target', '*.jpg')))

    image_dataset = ImageDataset(filesI, filesT)
    dataloader = DataLoader(image_dataset, batch_size=1, num_workers=1, shuffle=False)

    factor = 32
    PSNR, SSIM = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            imgI, imgT, filenames = batch
            imgI = imgI.to(device)

            h, w = imgI.shape[2], imgI.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)

            restored = model_restoration(imgI)[-1]
            restored = torch.clamp(restored, 0, 1)

            imgT = imgT.to(device)
            
            psnr_val = 10 * torch.log10(1 / F.mse_loss(restored[0].squeeze(0).cpu(), imgT.squeeze(0).cpu()))
                
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(F.adaptive_avg_pool2d(restored, (int(H / down_ratio), int(W / down_ratio))),
                            F.adaptive_avg_pool2d(imgT, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False).cpu()
            
            PSNR.append(psnr_val)
            SSIM.append(ssim_val)

            if args.save_images:
                os.makedirs(result_dir, exist_ok=True)
                save_img((os.path.join(result_dir, filenames[0])),
                        img_as_ubyte(restored[0].cpu().permute(1, 2, 0).numpy()))
                
    PSNR, SSIM = np.array(PSNR), np.array(SSIM)
    print("{}: PSNR {:4f} SSIM {:4f}".format(str(dataset), np.mean(PSNR), np.mean(SSIM)))


if __name__ == '__main__':
    main()
