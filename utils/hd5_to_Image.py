import h5py
from PIL import Image
import numpy as np
import json


filename = r"C:\Users\Lenna\Google Drive\Projektarbeit_ResearchProject\Github\HDF5_files\['kr', 'sb']_anno_anno_var_no1_rl_calcium_nocalcium_cartesian.h5"
image_dir = r"C:\Users\Lenna\Google Drive\Projektarbeit_ResearchProject\datasets\IVUS\images"
json_file_dir = r"C:\Users\Lenna\Google Drive\Projektarbeit_ResearchProject\datasets\IVUS"

def img_to_folder(image, img_name, dir):
    img = Image.fromarray(image)
    img.save(f'{dir}/{img_name}.png')

def json_list_append(label, labels, img_name):
    label_list = [f'images/{img_name}', label]
    labels.append(label_list)

def create_json_file(labels, json_file_dir):
    d = {"labels" : labels}
    with open(f'{json_file_dir}/dataset.json', 'w') as fp:
        json.dump(d, fp)


if __name__ == "__main__":
    labels = []
    counter = 0
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
                            img_name = f'{gname}_{sub_group_name}_{i}_' + '{0:05d}'.format(counter) + ".png"
                            img_to_folder(image, img_name, image_dir)
                            json_list_append(label, labels, img_name)
                            create_json_file(labels, json_file_dir)
                            counter = counter + 1