from random import sample
from re import I
import torch
from torch import nn
import albumentations

from collections import defaultdict
import pprint


import argparse
import json

import imgaug.augmenters as iaa
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils import data
from HDF5Dataset import HDF5Dataset
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from scipy.spatial import distance
import pandas as pd
from skimage.util import img_as_float
import csv
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

from torch.nn import functional as F

from torch.autograd import Variable

import os
import shutil
import tqdm


def augment_label_map(label_maps, display=False, output_folder=None, i=None):
        """
        Takes in a torch.tensor label map of shape: either (B x C x H x W) or (C x H x W) and outputs the transformed label map with same shape as input
        Additionally if display=True, the pixel values (classes) are changed such the different shapes are visible
        Additionally a output_folder can be set to save the input label_map and output_label map as png --> works only if a single label_map is given as input
        
        
        Possible augmentations from import imgaug.augmenters as iaa(https://github.com/aleju/imgaug):
        """

        seq = iaa.Sequential([
                # iaa.Sometimes(
                # 1,
                # iaa.ElasticTransformation(alpha=(250.0), sigma=50.),
                # ),
                iaa.Sometimes(
                1,
                iaa.CropAndPad(percent=(-0.10, 0.10)),
                ),
                iaa.Sometimes(
                1,
                iaa.PerspectiveTransform(scale=(0.01, 0.065)),
                ),
                iaa.Sometimes(
                1,
                iaa.geometric.Affine(rotate=(-15,15))
                ),
                iaa.Fliplr(0.5),
        ])

        if len(label_maps.shape) < 4:
                label_map_np = label_maps.numpy().transpose(1,2,0).astype(dtype=np.uint8)
                label_map_np_aug = seq(image=label_map_np)

                print(np.unique(label_map_np))
                print(np.unique(label_map_np_aug))
        else:
                label_maps_np = label_maps.numpy().transpose(0,2,3,1).astype(dtype=np.uint8)

                label_maps_np_aug = seq(images=label_maps_np)

        if display:

                label_map_np = np.squeeze(label_map_np, axis=2)

                label_map_np_aug = np.squeeze(label_map_np_aug, axis=2)

                label_map_np[label_map_np==3] = 255
                label_map_np[label_map_np==2] = 150
                label_map_np[label_map_np==1] = 50

                label_map_np_aug[label_map_np_aug==3] = 255
                label_map_np_aug[label_map_np_aug==2] = 150
                label_map_np_aug[label_map_np_aug==1] = 50

        if output_folder is not None:
                
                label_map_pil = Image.fromarray(label_map_np).convert("L")

                label_map_pil.save(os.path.join(output_folder, "test_no_augs", f"label_map_{i}.png"))

                label_map_aug_pil = Image.fromarray(label_map_np_aug).convert("L")

                label_map_aug_pil.save(os.path.join(output_folder, "test_augs", f"label_map_aug_{i}.png"))

                return

        label_maps_np_aug= label_maps_np_aug.transpose(0,3,1,2)

        return torch.tensor(label_maps_np_aug)


if __name__ == "__main__":

        GAN_models = ["StyleGAN2_DiffAug"]

        dataset_sizes = ["104", "42"]

        percentages = ["200perc", "100perc", "50perc"]
        
        counter = 0
        for GAN_model in GAN_models:
                for dataset_size in dataset_sizes:
                        for percentage in percentages:

                                train_dir = rf"G:\My Drive\Projektarbeit_ResearchProject\datasets\IVUS\IVUS_resized\train_val\260\{dataset_size}"
                                fake_dir = rf"G:\My Drive\Projektarbeit_ResearchProject\datasets\IVUS\IVUS_resized\fakes\{GAN_model}\{dataset_size}\{percentage}"
                                fakes_and_train_dir = rf"G:\My Drive\Projektarbeit_ResearchProject\datasets\IVUS\IVUS_resized\fakes_and_train\{GAN_model}\{dataset_size}\{percentage}"

                                print()
                                print(train_dir)
                                print(fake_dir)
                                print(fakes_and_train_dir)

                                a = pd.read_csv(os.path.join(train_dir, "train.csv"))
                                b = pd.read_csv(os.path.join(fake_dir, "train.csv"))
                                concat = pd.concat([a,b])
                                os.makedirs(os.path.join(fakes_and_train_dir, "images"), exist_ok=True)
                                concat.to_csv(os.path.join(fakes_and_train_dir, "train.csv"), index=False)

                                #copy files
                                os.makedirs(os.path.join(fakes_and_train_dir, "images"), exist_ok=True)
                                dest = os.path.join(fakes_and_train_dir, "images")

                                # copy train files
                                train_dir_imgs = os.listdir(os.path.join(train_dir, "images"))
                                for img in tqdm.tqdm(train_dir_imgs, desc = 'dirs'):
                                        full_file_name = os.path.join(os.path.join(train_dir, "images"), img)
                                        shutil.copy(full_file_name, dest)

                                # copy fake files
                                fake_dir_imgs = os.listdir(os.path.join(fake_dir, "images"))
                                for fake_img in tqdm.tqdm(fake_dir_imgs, desc = 'dirs'):
                                        full_file_name = os.path.join(os.path.join(fake_dir, "images"), fake_img)
                                        shutil.copy(full_file_name, dest)

                                counter += 1

        print(counter)