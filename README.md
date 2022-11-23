# GANs-Pytorch <!-- omit in toc -->

This repository contains different Generative adversarial network (GAN) architectures and functions for dataloading, training and evaluating GANs
There are two main folders:
1. basic_GANs
   - contains architectures, dataloaders, metrics and other utils to train basic GANs such as Deep Convolutional GAN
   - refer to train_DCGAN.py to start training 
2. StyleGAN2
   - contains code from [Differentiable Augmentation for Data-Efficient GAN Training](https://github.com/mit-han-lab/data-efficient-gans)
   - refer to section [Train StyleGAN2](##TrainStyleGAN2) to get information on how to train a basic StyleGAN2, StyleGAN2 with DiffAugment or StyleGAN2ADA (external link)


## Installation

- clone this repository:

```
git clone https://github.com/lennart-maack/GANs-Pytorch.git
cd GANs-Pytorch
```

## Dataset

You should structure your dataset in the following way:

```

dataset/
    ├── images/
        ├── name_of_image_1.png
        ├── name_of_image_2.png
        ...
    ├── dataset.json
```

- dataset.json should have the following format:
- class has to be set to the class index (Either 0 or 1 or .. or n_classes-1) of the according name_of_image_i.png

```
{"labels": 
    [["name_of_image_1.png", class],
     ["name_of_image_2.png", class],
     ...
    ]
}
```


## Train StyleGAN2
- for more in depth information about training options refer to StyleGAN2/train.py line 416 - line 455

### Train basic StyleGAN2

```
--data: Where to save your 
--outdir: Where to save your results (including sample imgs, logs etc.)
--cfg: set the config (cfg) to stylegan2 to use a basic stylegan2 without DiffAugment
--cond: conditional GAN, True or False
--metrics: Metrics to log during training (fid50k_full is best option)
--batch: batch size to train with
```


```
cd StyleGAN2
python3 train.py --data /path/to/dataset --outdir /path/to/outdir --cfg stylegan2 --cond True --metrics 'fid50k_full' --batch 32
```

### Train StyleGAN2 DiffAugment

```
--data: Where to save your 
--outdir: Where to save your results (including sample imgs, logs etc.)
--DiffAugment: Set the augmentations for training the StyleGAN2, options: "color","translation","cutout"
--cond: conditional GAN, True or False
--metrics: Metrics to log during training (fid50k_full is best option)
--batch: batch size to train with
```


```
cd StyleGAN2
python3 train.py --data /path/to/dataset --outdir /path/to/outdir --DiffAugment "color","translation","cutout" --cond True --metrics 'fid50k_full' --batch 32
```

### Train StyleGAN2 ADA

- Refer to the following repository: [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)



## How to solve specific issues

### Semaphore Problem

- To avoid the semaphore problem:

```
pip install ninja
pip install torch==1.8.1 torchvision==0.9.1
```