import h5py
from PIL import Image
import numpy as np
import json
import cv2
import os

# Set path to images for json file
image_path = "images"

# Resize to (512, 512)
resize_512 = True

# Set directories
filename = r"C:\Users\Lenna\Google Drive\Projektarbeit_ResearchProject\Github\HDF5_files\['kr', 'sb']_anno_anno_var_no1_rl_calcium_nocalcium_cartesian.h5"

train_dir = r"C:\Users\Lenna\Google Drive\Projektarbeit_ResearchProject\datasets\IVUS\IVUS_training_set_resized\images"
test_dir = r"C:\Users\Lenna\Google Drive\Projektarbeit_ResearchProject\datasets\IVUS\IVUS_test_set_resized\images"

train_json_file_dir = r"C:\Users\Lenna\Google Drive\Projektarbeit_ResearchProject\datasets\IVUS\IVUS_training_set_resized"
test_json_file_dir = r"C:\Users\Lenna\Google Drive\Projektarbeit_ResearchProject\datasets\IVUS\IVUS_test_set_resized"

test_patients = ["patient016", "patient019", "patient023", "patient025", "patient026", "patient034", "patient035", "patient044", "patient052", "patient064"]

def img_to_folder(image, img_name, dir):
    if resize_512:
        res = cv2.resize(image, dsize=(512,512))
    img = Image.fromarray(res.astype(np.uint8))
    img.save(f'{dir}/{img_name}')

def test_json_list_append(label, test_labels, img_name):
    test_label_list = [f'{image_path}/{img_name}', label] #wichtig damit der path name passt wenn resized!!
    test_labels.append(test_label_list)

def train_json_list_append(label, train_labels, img_name):
    train_label_list = [f'{image_path}/{img_name}', label] #wichtig damit der path name passt wenn resized!!
    train_labels.append(train_label_list)

def test_create_json_file(labels, test_json_file_dir):
    d = {"labels" : labels}
    with open(f'{test_json_file_dir}/dataset.json', 'w') as fp:
        json.dump(d, fp)

def train_create_json_file(labels, train_json_file_dir):
    d = {"labels" : labels}
    with open(f'{train_json_file_dir}/dataset.json', 'w') as fp:
        json.dump(d, fp)


if __name__ == "__main__":
    test_labels = []
    train_labels = []
    counter = 0
    test_counter = 0
    train_counter = 0
    with h5py.File(filename) as h5_file:
        # Walk through all groups, extracting datasets
        for gname, group in h5_file.items():
            for sub_group_name, sub_group in group.items():
                for dname, ds in sub_group.items():
                    if dname == "images":
                        for i in range(len(ds)):
                            if group[sub_group_name]["labels"][i][0]:
                                label = 1
                            else:
                                label = 0
                            image = group[sub_group_name]["images"][i]
                            img_name = f'{gname}_{sub_group_name}_{i}_' + '{0:05d}'.format(counter) + '.png'
                            if gname in test_patients:
                                test_counter = test_counter + 1
                                img_to_folder(image, img_name, test_dir)
                                test_json_list_append(label, test_labels, img_name)
                                test_create_json_file(test_labels, test_json_file_dir)
                            else:
                                train_counter = train_counter + 1
                                img_to_folder(image, img_name, train_dir)
                                train_json_list_append(label, train_labels, img_name)
                                train_create_json_file(train_labels, train_json_file_dir)
                            counter = counter + 1
    print("test: ", test_counter)
    print("train: ", train_counter)