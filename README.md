# Residual U-Net with Attention Mechanism

Medical Image Segmentation using Residual U-Net with Attention Mechanism.

The network takes advantage of residual blocks, atrous spatial pyramid pooling, and channel- and spatial-attention blocks. We take ResNet50 as backbone to facilitate the training process.

## Network architecture

<p align="center">
    <img src="img/block_diagram.png" width="50%"/>
</p>
<h6 align="center">Block diagram</h6>

<p align="center">
    <img src="img/3D_arch.png" width="85%"/>
</p>
<h6 align="center">3D architecture</h6>

## Datasets

The following publicly available datasets are used in the experiments:

| Dataset                                                      | Number of mages | Image shape (W x H) |
| ------------------------------------------------------------ | --------------- | ------------------- |
| [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/) | 612             | 384 x 288           |
| [CVC-ColonDB](https://drive.google.com/file/d/1S0GvCLOoSiePEiJJkX3r-RCd34PHFyQF/view?usp=sharing) | 380             | 574 x 500           |
| [ETIS-LaribPolypDB](https://polyp.grand-challenge.org/EtisLarib/) | 196             | 1225 x 966          |
| [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)         | 1000            | Variable            |

## Image preprocessing

The following transformations are done with the help of [albumentations](https://github.com/albumentations-team/albumentations):

- Crop, flip, rotate, transpose
- Random brightness contrast
- Random Gamma
- Hue Saturation Value (HSV)
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Blur (motion blur, median blur, Gaussian blur)
- Gauss noise
- RGB shift
- Channel shuffle
- Coarse dropout
- ...

After these operations, the number of training samples increased by 25 times.

## Hyperparameters

- Bacth size: 4
- Epochs: 250
- Learning rate: 1e-5
- Optimizer: Nadam
- Loss: Dice loss

## Configurations for CUDA and cuDNN

- Compatible versions of tensorflow-gpu, python, CUDA, and cuDNN: [Tested build configurations](https://tensorflow.google.cn/install/source_windows#gpu)
- CUDA Toolkit and corresponding driver versions: [NVIDIA CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#abstract)
- Driver download: [NVIDIA Driver Downloads](http://www.nvidia.com/drivers)
- CUDA download: [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- cuDNN download: [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

The latest tensorflow-gpu solved [this bug](https://github.com/tensorflow/tensorflow/issues/46249), so I just switch to tf2.6 with CUDA 11.2 and cuDNN 8.1 installed.

