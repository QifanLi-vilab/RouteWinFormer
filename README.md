

## RouteWinFormer: A Mid-range Route-Window Transformer with Structure Regularization for Image Restoration



#### Qifan Li, Tianyi Liang, Binbin Song, Xingtao Wang, Yi Zheng, Jisheng Chu, Xiaopeng Fan


### Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks

```python
python 3.9.18
pytorch 2.1.0
cuda 12.1
```

```
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Test Instruction
* Image Defocus Deblurring:
```
    cd test_defocus_deblur
    python python test_defocus_deblur.py --input_dir Path_to_Dir/Datasets/DPDD/test/ --dim 32 --weights ../Pretrained/DPDD32.pth --save_images
    python python test_defocus_deblur.py --input_dir Path_to_Dir/Datasets/DPDD/test/ --dim 64 --weights ../Pretrained/DPDD64.pth --save_images
```
* Image DeSnowing:

  * CSD
      ```
      cd test_desnowing
      python test_CSD.py --input Path_to_Dir/Datasets/Desnowing/CSD/test2000 --weights ../Pretrained/CSD.pth --save_images
      ```
  * SRRS
      ```
      cd test_desnowing
      python test_SRRS.py --input Path_to_Dir/Datasets/Desnowing/SRRS/test2000 --weights ../Pretrained/SRRS.pth --save_images
      ```
    
* Image DeHazing:

  * Haze4K
      ```
      cd test_dehazing
      python test_haze4k.py --input_dir Path_to_Dir/Datasets/Dehazing/ --weights ../Pretrained/Haze4K.pth --save_images
      ```
  * NHHAZE
      ```
      cd test_dehazing
      python test_nhhaze.py --input_dir Path_to_Dir/Datasets/Dehazing/ --weights ../Pretrained/NHHAZE.pth
      ```
* Image DeRaining:

  * Rain100L/Test1200
      ```
      cd test_dehazing
      python test_derain.py --input_dir Path_to_Dir/Datasets/Deraining/Rain/test/ --weights ../Pretrained/DeRain.pth
      ```
* Image Motion Deblurring:

  * GoPro
      ```
      cd test_dehazing
      python test_GoPro.py --input_dir Path_to_Dir/Datasets/GoPro/test/ --weights ../Pretrained/GoPro.pth --save_images
      ```
  * HIDE
      ```
      cd test_dehazing
      python test_HIDE.py --input_dir Path_to_Dir/Datasets/GoPro/test/ --weights ../Pretrained/GoPro.pth --save_imag
      ```

### Pre-trained Models |  [百度网盘](https://pan.baidu.com/s/1ChCiMJ_zESaV1_dliezApQ?pwd=pngt) 提取码: pngt




