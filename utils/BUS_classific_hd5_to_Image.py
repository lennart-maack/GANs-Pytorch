import h5py
from PIL import Image
import numpy as np
import json
import cv2
import os
import pandas as pd

import BUS_sets


def save_img(img_array, resize, save_path, filename):
    """
    Saves an image with given size and filename into a given path
    resize: tuple of size of wanted size of image
    """


    if resize is not None:
        img_array = cv2.resize(img_array, dsize=resize)

    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(os.path.join(f"{save_path}/images", filename))


def build_json_entry(filename, label):

    single_entry = [f'images/{filename}', label]
    return single_entry


def export_json(json_list, path):

    d = {"labels":json_list}
    with open(os.path.join(path, "dataset.json"), 'w') as fp:
        json.dump(d, fp)



def save_img_and_json(h5_path, resize, save_path, test_set, train_val_set):

    """
    Reads, resizes and saves the images from the h5 file (h5_path) to path/image with specific filename
    The filename consits of the class (benign, malignant or normal) and the patient number.
    Additionally a corresponding json file (in the format for STYLEGAN2 ADA) is saved to path
    """

    json_list = []

    with h5py.File(h5_path) as h5_file:
        for gname, group in h5_file.items():
            for patient_id, patient_content in group.items():
                for item_name, item in patient_content.items():

                    filename = f"{gname}_{patient_id}_{item_name}.png"
        
                    if item_name == "image":

                        if gname == "benign":
                            if int(patient_id) in test_set[0]:
                                continue
                            if int(patient_id) not in train_val_set[0]:
                                continue
                            label=0

                        elif gname == "malignant":
                            if int(patient_id) in test_set[1]:
                                continue
                            if int(patient_id) not in train_val_set[1]:
                                continue
                            label=1

                        elif gname == "normal":
                            if int(patient_id) in test_set[2]:
                                continue
                            if int(patient_id) not in train_val_set[2]:
                                continue
                            label=2

                        img_array = np.zeros(item.shape, dtype="uint8")
                        item.read_direct(img_array)
                        
                        #save_img(img_array, resize, save_path, filename)

                        single_entry = build_json_entry(filename, label)
                        json_list.append(single_entry)
        
    print(len(json_list))
    export_json(json_list, save_path)                    


def save_img_and_csv(h5_path, resize, test_path, train_val_path, test_set, train_val_set):


    test_df = pd.DataFrame(columns = ["image", "target"])
    train_df = pd.DataFrame(columns = ["image", "target", "kfold"])
    with h5py.File(h5_path) as h5_file:
        for gname, group in h5_file.items():
                for patient_id, patient_content in group.items():
                    for item_name, item in patient_content.items():

                        filename = f"{gname}_{patient_id}_{item_name}.png"

                        if item_name == "image":

                            img_array = np.zeros(item.shape, dtype="uint8")
                            item.read_direct(img_array)

                            if gname == "benign":

                                if int(patient_id) in test_set[0]:
                                    test_df = test_df.append({"image" : filename}, ignore_index=True)
                                    test_df.loc[test_df["image"]== filename, "target"] = 0
                                    #save_img(img_array, resize, test_path, filename)
                                    continue

                                for kfold, fold in enumerate(train_val_set[0]):
                                    if int(patient_id) not in fold:
                                        continue
                                    train_df = train_df.append({"image" : filename}, ignore_index=True)
                                    train_df.loc[train_df["image"]==filename, "target"] = 0
                                    train_df.loc[train_df["image"]==filename, "kfold"] = kfold
                                    #save_img(img_array, resize, train_val_path, filename)

                            if gname == "malignant":

                                if int(patient_id) in test_set[1]:
                                    test_df = test_df.append({"image" : filename}, ignore_index=True)
                                    test_df.loc[test_df["image"]== filename, "target"] = 1
                                    #save_img(img_array, resize, test_path, filename)
                                    continue

                                for kfold, fold in enumerate(train_val_set[1]):
                                    if int(patient_id) not in fold:
                                        continue
                                    train_df = train_df.append({"image" : filename}, ignore_index=True)
                                    train_df.loc[train_df["image"]==filename, "target"] = 1
                                    train_df.loc[train_df["image"]==filename, "kfold"] = kfold
                                    #save_img(img_array, resize, train_val_path, filename)

                            
                            if gname == "normal":

                                if int(patient_id) in test_set[2]:
                                    test_df = test_df.append({"image" : filename}, ignore_index=True)
                                    test_df.loc[test_df["image"]== filename, "target"] = 2
                                    #save_img(img_array, resize, test_path, filename)
                                    continue

                                for kfold, fold in enumerate(train_val_set[2]):
                                    if int(patient_id) not in fold:
                                        continue
                                    train_df = train_df.append({"image" : filename}, ignore_index=True)
                                    train_df.loc[train_df["image"]==filename, "target"] = 2
                                    train_df.loc[train_df["image"]==filename, "kfold"] = kfold
                                    #save_img(img_array, resize, train_val_path, filename)

    #test_df.to_csv(os.path.join(test_path, "test.csv"), index=False, header=True)
    #train_df.to_csv(os.path.join(train_val_path, "train.csv"), index=False, header=True)


def main():

    h5_path = r"G:\My Drive\Projektarbeit_ResearchProject\Github\HDF5_files\US_breast_cancer.h5"

    #Test set

    test_set = BUS_sets.get_BUS_test()

    BUS_500 = BUS_sets.get_BUS_500()

    #BUS_sets.check_length(BUS_500)

    #print()
    BUS_250 = BUS_sets.get_BUS_250()

    #BUS_sets.check_length(BUS_250)

    #print()
    BUS_100 = BUS_sets.get_BUS_100()

    #BUS_sets.check_length(BUS_100)

    #print()
    BUS_50 = BUS_sets.get_BUS_50()

    #BUS_sets.check_length(BUS_50)

    resize=(260,260)

    train_val_path = r"G:\My Drive\Projektarbeit_ResearchProject\datasets\BUS_classification\resized\train_val\260\50"

    test_path = r"G:\My Drive\Projektarbeit_ResearchProject\datasets\BUS_classification\resized\test\260"

    #save_img_and_json(h5_path, resize, save_path, test_set, train_val_set=train_val_100)
    save_img_and_csv(h5_path, resize, test_path, train_val_path, test_set, train_val_set=BUS_50)


if __name__ == "__main__":

    main()