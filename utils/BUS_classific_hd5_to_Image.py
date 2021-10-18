import h5py
from PIL import Image
import numpy as np
import json
import cv2
import os
import pandas as pd



def save_img(img_array, resize, save_path, filename):

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


if __name__ == "__main__":


    h5_path = r"G:\My Drive\Projektarbeit_ResearchProject\Github\HDF5_files\US_breast_cancer.h5"

    save_path = r""

    resize = (256,256)

    test_set_b =  [409, 133, 231, 424, 40, 165, 18, 116, 324, 122, 276, 270, 266, 404, \
        161, 305, 176, 434, 430, 403, 14, 402, 67, 293, 238, 290, 63, 334, 69, 265, 321, \
        132, 42, 220, 381, 135, 304, 90, 60, 388, 337, 185, 411, 92, 5, 384, 269, 174, \
        397, 125, 119, 37, 310, 47, 234, 253, 209, 53, 54, 175, 226, 134, 82, 247, 153, 219, \
        99, 244, 38, 375, 407, 30, 340, 192, 28, 33, 156, 44, 103, 6, 77, 278, 12, 281, 354, \
        398, 195, 256, 360, 355, 251, 314, 224, 299, 383, 401, 114, 250, 70, 164, 273, 306, \
        277, 131, 289, 34, 333, 106, 395, 279, 353, 435, 339, 27, 318, 315, 410, 336, 138, 184, \
        396, 168, 193, 374, 217, 50, 71, 372, 128, 16, 205, 191, 294, 147, 309, 113, 259, 280, 323, \
        24, 376, 233, 80, 142, 377, 389, 23, 349, 26, 414, 262, 284, 386, 166, 57, 212, 177, 35]

    test_set_m = [15, 185, 170, 165, 78, 22, 52, 3, 39, 192, 6, 141, 201, 98, 128, 126, \
        112, 150, 163, 177, 198, 125, 35, 82, 137, 153, 139, 66, 145, 17, 53, 172, 191, \
        208, 14, 159, 158, 184, 168, 69, 92, 84, 97, 109, 25, 36, 64, 8, 103, 96, 147, 74, \
        155, 57, 130, 140, 162, 200, 188, 117, 40, 71, 189, 195, 12, 175, 13, 131, 142, 193, \
        83, 29, 106, 89, 18]

    test_set_n = [102, 117, 132, 53, 85, 7, 80, 79, 109, 84, 35, 46, 99, 42, 49, 120, 55, \
        128, 133, 93, 32, 4, 52, 48, 74, 37, 78, 110, 130, 65, 70, 81, 126, 94, 59, 20, 36, \
        95, 98, 6, 88, 13, 83, 29, 106, 89, 18]

    test_set = [test_set_b,test_set_m, test_set_n]


    train_val_100_b = [137, 1, 97, 428, 65, 379, 257, 189, 218, 307, 416, 312, 111, 356, 201, 241, \
        358, 20, 155, 243, 292, 400, 112, 8, 302, 267, 431, 258, 390, 141, 308, 163, 94, 291, 272, 108, \
        367, 117, 196, 335, 420, 75, 329, 46, 15, 399, 425, 433, 369, 213, 249, 170, 173, 78, 4, 350]

    train_val_100_m = [144, 43, 186, 28, 167, 157, 62, 121, 46, 99, 70, 181, 116, 58, 204, 105, 16, 199, \
        210, 67, 114, 32, 30, 206, 10, 174, 87]

    train_val_100_n = [67, 23, 25, 51, 44, 12, 129, 107, 86, 38, 92, 43, 61, 24, 101, 64, 76]

    train_val_100 = [train_val_100_b, train_val_100_m, train_val_100_n]

    #save_img_and_json(h5_path, resize, save_path, test_set, train_val_set=train_val_100)

    print("finished")