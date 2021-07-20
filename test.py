import argparse
import os

import PIL
import torch
import torchvision
from tqdm import tqdm
from torch.utils import tensorboard
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils


import torch_fidelity

parser = argparse.ArgumentParser()

parser.add_argument("--print_name", type=str, default="default", help="message to print")
opt = parser.parse_args()
print(opt)


test_list = []
for i in tqdm(range(100000)):
    test_list.append(i)
    if i % 1000 == 0:
        print(test_list[i])
        print(opt.print_name)

input("Press enter to exit")
