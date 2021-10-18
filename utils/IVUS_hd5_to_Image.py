import h5py
from PIL import Image
import numpy as np
import json
import cv2
import os
import pandas as pd

# Set directories
filename = r"G:\My Drive\Projektarbeit_ResearchProject\Github\HDF5_files\['kr', 'sb']_anno_anno_var_no1_rl_calcium_nocalcium_cartesian.h5"

def img_to_folder(image, img_name, dir, resize):
    if resize:
        res = cv2.resize(image, dsize=(260,260))
    img = Image.fromarray(res.astype(np.uint8))
    img.save(f'{dir}/images/{img_name}')

def test_json_list_append(label, test_labels, img_name):
    test_label_list = [f'images/{img_name}', label]
    test_labels.append(test_label_list)

def train_json_list_append(label, train_labels, img_name):
    train_label_list = [f'images/{img_name}', label] #wichtig damit der path name passt wenn resized!!
    train_labels.append(train_label_list)

def test_create_json_file(labels, test_json_file_dir):
    d = {"labels" : labels}
    with open(f'{test_json_file_dir}/dataset.json', 'w') as fp:
        json.dump(d, fp)

def train_create_json_file(labels, train_json_file_dir):
    d = {"labels" : labels}
    with open(f'{train_json_file_dir}/dataset.json', 'w') as fp:
        json.dump(d, fp)



def h5py_to_image_json(test_patients, leave_out_patients, resize, test_dir, test_json_file_dir, train_dir, train_json_file_dir):

    test_labels = []
    train_labels = []
    counter = 0
    test_counter = 0
    train_counter = 0
    leave_out_counter = 0
    positive_counter = 0
    negative_counter = 0
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
                                if test_dir is None:
                                    continue
                                test_counter = test_counter + 1
                                img_to_folder(image, img_name, test_dir, resize)
                                test_json_list_append(label, test_labels, img_name)
                                test_create_json_file(test_labels, test_json_file_dir)
                            elif leave_out_patients is not None and gname in leave_out_patients:
                                leave_out_counter = leave_out_counter + 1
                            else:
                                train_counter = train_counter + 1
                                img_to_folder(image, img_name, train_dir, resize)
                                train_json_list_append(label, train_labels, img_name)
                                train_create_json_file(train_labels, train_json_file_dir)
                            counter = counter + 1
    print("test: ", test_counter)
    print("train: ", train_counter)
    print("leave out: ", leave_out_counter)

def h5py_to_image_csv(test_patients, cv_sets, test_dir, train_dir):
    
    #set values
    counter = 0
    test_counter = 0
    train_counter = 0
    test_df = pd.DataFrame(columns = ["image", "target"])
    train_df = pd.DataFrame(columns = ["image", "target", "kfold"])

    with h5py.File(filename) as h5_file:
        # Walk through all groups, extracting datasets
        for gname, group in h5_file.items():
            for sub_group_name, sub_group in group.items():
                for dname, ds in sub_group.items():
                    if dname == "images":
                        for i in range(len(ds)):
                            image = group[sub_group_name]["images"][i]
                            img_name = f'{gname}_{sub_group_name}_{i}_' + '{0:05d}'.format(counter) + '.png'
                            
                            if gname in test_patients:
                                
                                img_to_folder(image, img_name, test_dir, resize=True)

                                test_df.at[test_counter, "image"] = img_name
                                if group[sub_group_name]["labels"][i][0]:
                                    test_df.at[test_counter, "target"] = 1
                                else:
                                    test_df.at[test_counter, "target"] = 0
                                test_counter = test_counter + 1
                                
                            else:
                                
                                img_to_folder(image, img_name, train_dir, resize=True)

                                train_df.at[train_counter, "image"] = img_name
                                if group[sub_group_name]["labels"][i][0]:
                                    train_df.at[train_counter, "target"] = 1
                                else:
                                    train_df.at[train_counter, "target"] = 0

                                if gname in cv_sets[0]:
                                    train_df.at[train_counter, "kfold"] = 0
                                elif gname in cv_sets[1]:
                                    train_df.at[train_counter, "kfold"] = 1
                                elif gname in cv_sets[2]:
                                    train_df.at[train_counter, "kfold"] = 2

                                train_counter = train_counter + 1
                                
                            counter = counter + 1
    
    print("test: ", test_counter)
    print("train: ", train_counter)
    print("counter all: ", counter)

    test_df.to_csv(os.path.join(test_dir, "test.csv"), index=False, header=True)
    train_df.to_csv(os.path.join(train_dir, "train.csv"), index=False, header=True)

def get_h5py_information(cv_sets, leave_out_patients):

    counter = 0
    test_counter = 0
    train_counter = 0
    leave_out_counter = 0
    positive_counter = 0
    negative_counter = 0
    positive_counter_cv0 = 0
    positive_counter_cv1 = 0
    positive_counter_cv2 = 0
    negative_counter_cv0 = 0
    negative_counter_cv1 = 0
    negative_counter_cv2 = 0
    error_counter = 0

    with h5py.File(filename) as h5_file:
        # Walk through all groups, extracting datasets
        for gname, group in h5_file.items():
            for sub_group_name, sub_group in group.items():
                for dname, ds in sub_group.items():
                    if dname == "images":
                        for i in range(len(ds)):
                            if group[sub_group_name]["labels"][i][0]:
                                label = 1
                                positive_counter = positive_counter + 1
                                if gname in cv_sets[0]:
                                    positive_counter_cv0 = positive_counter_cv0 + 1
                                elif gname in cv_sets[1]:
                                    positive_counter_cv1 = positive_counter_cv1 + 1
                                elif gname in cv_sets[2]:
                                    positive_counter_cv2 = positive_counter_cv2 + 1
                                else:
                                    error_counter = error_counter + 1
                            else:
                                label = 0
                                negative_counter = negative_counter + 1
                                if gname in cv_sets[0]:
                                    negative_counter_cv0 = negative_counter_cv0 + 1
                                elif gname in cv_sets[1]:
                                    negative_counter_cv1 = negative_counter_cv1 + 1
                                elif gname in cv_sets[2]:
                                    negative_counter_cv2 = negative_counter_cv2 + 1
                                else:
                                    error_counter = error_counter + 1

                            image = group[sub_group_name]["images"][i]
                            img_name = f'{gname}_{sub_group_name}_{i}_' + '{0:05d}'.format(counter) + '.png'
                            if gname in test_patients:
                                test_counter = test_counter + 1
                            elif leave_out_patients is not None and gname in leave_out_patients:
                                leave_out_counter = leave_out_counter + 1
                            else:
                                train_counter = train_counter + 1
                            counter = counter + 1

    print("test: ", test_counter)
    print("train: ", train_counter)
    print("leave out: ", leave_out_counter)
    print("positive: ", positive_counter)
    print("negative: ", negative_counter)
    print("positive_cv0: ", positive_counter_cv0)
    print("positive_cv1: ", positive_counter_cv1)
    print("positive_cv2: ", positive_counter_cv2)
    print("negative_cv0: ", negative_counter_cv0)
    print("negative_cv1: ", negative_counter_cv1)
    print("negative_cv2: ", negative_counter_cv2)
    print("errors: ", error_counter, "Needs to be equal to test size")

if __name__ == "__main__":
    
    # leave out patients
    patient = "patient0"

    numbers = [12, 17, 18, 24, 43, 55, \
                13, 22, 32, 42, 46, 60, \
                36, 40, 45, 48, 49, 50]
                #'{0:02d}'.format(7), 10, \
                #'{0:02d}'.format(9), 11, \
                #20, 27]
                #'{0:02d}'.format(5), \
                #'{0:02d}'.format(4), \
                #'{0:02d}'.format(14)]

    # test patients
    test_patients = ["patient016", "patient019", "patient023", "patient025", "patient026", "patient034", "patient035", "patient044", "patient052", "patient064"]
    
    leave_out_patients = [patient +  str(x) for x in numbers]

    cv_sets = [["patient003", "patient005", "patient007", "patient010", "patient012", "patient017", "patient018", "patient024", "patient043", "patient055"] , \
        ["patient002", "patient004", "patient009", "patient011", "patient013", "patient022", "patient032", "patient042", "patient046", "patient060"], \
        ["patient001", "patient014", "patient020", "patient027", "patient036", "patient040", "patient045", "patient048", "patient049", "patient050"]]

    #h5py_to_image_json(test_patients, leave_out_patients, resize = True)

    #get_h5py_information(cv_sets, leave_out_patients=None)

    dir_test = r"G:\My Drive\Projektarbeit_ResearchProject\datasets\IVUS\IVUS_resized\test\260"
    dir_train = r"G:\My Drive\Projektarbeit_ResearchProject\datasets\IVUS\IVUS_resized\train_val\260\25"

    dir_test_json = r"G:\My Drive\Projektarbeit_ResearchProject\datasets\IVUS\IVUS_resized\test\260"
    dir_train_json = r"G:\My Drive\Projektarbeit_ResearchProject\datasets\IVUS\IVUS_resized\train_val\260\25"

    #h5py_to_image_csv(test_patients, cv_sets, test_dir=dir_test, train_dir=dir_train)

    h5py_to_image_json(
        test_patients=test_patients,
        leave_out_patients=None,
        resize=True,
        test_dir=None,
        test_json_file_dir=dir_test_json,
        train_dir=dir_train,
        train_json_file_dir=dir_train_json)
