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
    parser.add_argument('--input_dir',default='/mnt/Data/qfl/Datasets/Dehazing/',type=str,
                        help='Directory of validation images')
    parser.add_argument('--result_dir', default='./', type=str, help='Directory for results')
    parser.add_argument('--weights',
                        default='../Pretrained/Haze4K.pth',
                        type=str,help='Path to pretrained weights')
    parser.add_argument('--dataset', default='Haze4K', type=str,
                        help='Test Dataset')
    parser.add_argument('--save_images', action='store_true', help='Save restored images')
    return parser.parse_args()

class ImageDataset(Dataset):

    def __init__(self, root_dir):
        self.input_dir = os.path.join(root_dir, 'input')
        self.target_dir = os.path.join(root_dir, 'target')

        self.input_files = natsorted(
            glob(os.path.join(self.input_dir, '*.png')) +
            glob(os.path.join(self.input_dir, '*.jpg'))
        )

        if len(self.input_files) == 0:
            raise RuntimeError(f'No input images found in {self.input_dir}')

    def __len__(self):
        return len(self.input_files)

    def _map_to_target(self, input_path):
        name = os.path.basename(input_path)
        stem, ext = os.path.splitext(name)

        base_id = stem.split('_')[0]
        target_name = base_id + ext
        target_path = os.path.join(self.target_dir, target_name)

        if not os.path.exists(target_path):
            raise FileNotFoundError(f'Missing target image: {target_path}')

        return target_path

    def __getitem__(self, idx):
        input_path = self.input_files[idx]
        target_path = self._map_to_target(input_path)

        imgI = np.float32(utils.load_img(input_path)) / 255.
        imgT = np.float32(utils.load_img(target_path)) / 255.

        imgI = torch.from_numpy(imgI).permute(2, 0, 1)
        imgT = torch.from_numpy(imgT).permute(2, 0, 1)

        filename = os.path.basename(target_path)

        return imgI, imgT, filename

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'Haze4K':
        yaml_file = '../options/test/DeHazing/RouteWinFormer-Haze4K-width48.yml'
    else:
        assert "dataset name error!"

    with open(yaml_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    cfg['network_g'].pop('type', None)

    model = RouteWinFormer(**cfg['network_g'])
    checkpoint = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(checkpoint['params'], strict=True)

    model = nn.DataParallel(model).to(device)
    model.eval()

    print(f'===> Loaded weights: {args.weights}')

    test_root = os.path.join(args.input_dir, args.dataset, 'test')
    
    image_dataset = ImageDataset(test_root)

    dataloader = DataLoader(
        image_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    result_dir = os.path.join(args.result_dir, args.dataset)
    os.makedirs(result_dir, exist_ok=True)

    factor = 32
    PSNR, SSIM = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            imgI, imgT, filenames = batch
            imgI = imgI.to(device)
            
            # used for SSIM ---> ConvIR
            h, w = imgI.shape[2], imgI.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
            
            restored = model(imgI)[-1]
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
    print("{}: PSNR {:4f} SSIM {:4f}".format(str(args.dataset), np.mean(PSNR), np.mean(SSIM)))


if __name__ == '__main__':
    main()

