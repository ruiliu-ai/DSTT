# Decoupled Spatial-Temporal Transformer for Video Inpainting

By [Rui Liu](https://ruiliu-ai.github.io), Hanming Deng, Yangyi Huang, Xiaoyu Shi, Lewei Lu, Wenxiu Sun, Xiaogang Wang, Jifeng Dai, Hongsheng Li. 

This repo is the official Pytorch implementation of [Decoupled Spatial-Temporal Transformer for Video Inpainting](https://arxiv.org/abs/2104.06637).

## Introduction
<img src='imgs/intro1.png' width="900px">
<div align=center><img src='imgs/intro2.png' width="350px"></div>

## Usage

### Prerequisites
- Python >= 3.6
- Pytorch >= 1.0 and corresponding torchvision (https://pytorch.org/)

### Install
- Clone this repo:
```
git clone https://github.com/ruiliu-ai/DSTT.git
```
- Install other packages:
```
cd DSTT
pip install -r requirements.txt
```

## Training

### Dataset preparation
Download datasets ([YouTube-VOS](https://competitions.codalab.org/competitions/19544) and [DAVIS](https://davischallenge.org/davis2017/code.html)) into the data folder.
```
mkdir data
```

### Training script
```
python train.py -c configs/youtube-vos.json
```

## Test
Download [pre-trained model](https://drive.google.com/file/d/1Fq3seV2X6dthbjdw4RTNyVd4HH2WlL7g/view?usp=sharing) into checkpoints folder.
```
mkdir checkpoints
```

### Test script
```
python test.py -c checkpoints/dstt.pth -v data/DAVIS/JPEGImages/blackswan -m data/DAVIS/Annotations/blackswan
```

## Citing DSTT
If you find DSTT useful in your research, please consider citing:
```
@article{Liu_2021_DSTT,
  title={Decoupled Spatial-Temporal Transformer for Video Inpainting},
  author={Liu, Rui and Deng, Hanming and Huang, Yangyi and Shi, Xiaoyu and Lu, Lewei and Sun, Wenxiu and Wang, Xiaogang and Li Hongsheng},
  journal={arXiv preprint arXiv:2104.06637},
  year={2021}
}
```

## Acknowledement
This code relies heavily on the video inpainting framework from [spatial-temporal transformer net](https://github.com/researchmm/STTN). 
