import h5py
from PIL import Image
import numpy as np
import json
import cv2
import os
import pandas as pd

import Heart_sets


def save_img(img_array, resize, save_path, filename):
    """
    Saves an image with given size and filename into a given path
    resize: tuple of size of wanted size of image
    """

    if resize is not None:
        img_array = cv2.resize(img_array, dsize=resize)

    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(os.path.join(f"{save_path}", filename))


def heart_h5py_to_image_csv(filename, resize, test_path, train_val_path, test_set, train_val_set):
    
    #set values
    counter = 0
    image_counter = 0
    mask_counter = 0
    test_counter = 0
    train_df = pd.DataFrame(columns = ["image", "kfold"])

    with h5py.File(filename) as h5_file:
        # Walk through all groups, extracting datasets
        for gname, group in h5_file.items():
            for subgroup_name, subgroup in group.items():
                for subsubgroup_name, subsubgroup in subgroup.items():
                    for item_name, item in subsubgroup.items():
                        
                        name = f"{gname}_{subgroup_name}_{subsubgroup_name}.png"
                        
                        patient_id = int(gname[-3:])

                        if patient_id in test_set:

                            if item_name == "image":
                                img_array = np.zeros(item.shape, dtype="uint8")
                                item.read_direct(img_array)
                                #save_img(img_array, resize, os.path.join(test_path, "images"), name)

                            if item_name == "mask":
                                img_array = np.zeros(item.shape, dtype="uint8")
                                item.read_direct(img_array)
                                #img_array[img_array==3] = 255
                                #img_array[img_array==2] = 150
                                #img_array[img_array==1] = 50
                                #save_img(img_array, resize, os.path.join(test_path, "masks"), name)

                            continue


                        if item_name == "image":

                            for fold, fold_items in enumerate(train_val_set):
                                if patient_id in fold_items:
                                    train_df = train_df.append({"image" : name}, ignore_index=True)
                                    train_df.loc[train_df["image"]==name, "kfold"] = fold
                                    img_array = np.zeros(item.shape, dtype="uint8")
                                    item.read_direct(img_array)
                                    save_img(img_array, resize, os.path.join(train_val_path, "images"), name)
                                    image_counter += 1


                        if item_name == "mask":

                            for fold, fold_items in enumerate(train_val_set):
                                if patient_id in fold_items:
                                    img_array = np.zeros(item.shape, dtype="uint8")
                                    item.read_direct(img_array)
                                    #img_array[img_array==3] = 255
                                    #img_array[img_array==2] = 150
                                    #img_array[img_array==1] = 50
                                    save_img(img_array, resize, os.path.join(train_val_path, "masks"), name)
                                    mask_counter += 1
                        
    print(image_counter, mask_counter)
    train_df.to_csv(os.path.join(train_val_path, "gt.csv"), index=False, header=True)                    

if __name__ == "__main__":


    # Set directory
    h5_path = r"C:\Users\Lenna\Documents\TUHH Master\SoSe21\Research Project\datasets\MedicalData\DATASETS\US_Heart_Dataset\data\US_heart.h5"

    test_path = r"G:\My Drive\Projektarbeit_ResearchProject\datasets\Heart_seg\resized\test\256"

    train_val_path = r"G:\My Drive\Projektarbeit_ResearchProject\datasets\Heart_seg\resized\train_val\256\20"


    test_set = Heart_sets.get_heart_test()

    heart_set = Heart_sets.get_heart_20()

    print(len(test_set))
    
    length = 0
    for i, item in enumerate(heart_set):
        length += len(heart_set[i])
        print(len(heart_set[i]))
    print(length)

    resize = (256,256)

    heart_h5py_to_image_csv(h5_path, resize, test_path, train_val_path, test_set, train_val_set=heart_set)