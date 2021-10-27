import h5py
from PIL import Image
import numpy as np
import json
import cv2
import os
import pandas as pd



def img_to_folder(image, img_name, dir, resize):
    if resize:
        res = cv2.resize(image, dsize=(256,256))
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

def h5py_to_image_json(test_patients, cv_sets, resize, test_dir, train_dir):

    test_labels = []
    train_labels = []
    counter = 0
    test_counter = 0
    train_counter_0 = 0
    train_counter_1 = 0
    train_counter_2 = 0
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

                            counter += 1
                            #print("ds: ", ds)
                            if group[sub_group_name]["labels"][i][0]:
                                label = 1
                                positive_counter += 1
                            else:
                                label = 0
                                negative_counter += 1
                            image = group[sub_group_name]["images"][i]
                            img_name = f'{gname}_{sub_group_name}_{i}_' + '{0:05d}'.format(counter) + '.png'
                            
                            if gname in test_patients:
                                if test_dir is None:
                                    continue
                                test_counter = test_counter + 1
                                #img_to_folder(image, img_name, test_dir, resize)
                                #test_json_list_append(label, test_labels, img_name)
                                #test_create_json_file(test_labels, test_dir)
                                continue
                            
                            if gname in cv_sets[0] and sub_group_name == "pullback01":
                                train_counter_0 = train_counter_0 + 1
                                img_to_folder(image, img_name, train_dir, resize)
                                train_json_list_append(label, train_labels, img_name)
                                train_create_json_file(train_labels, train_dir)
                
                            if gname in cv_sets[1] and sub_group_name == "pullback01":
                                train_counter_1 = train_counter_1 + 1
                                img_to_folder(image, img_name, train_dir, resize)
                                train_json_list_append(label, train_labels, img_name)
                                train_create_json_file(train_labels, train_dir)

                            if gname in cv_sets[2] and sub_group_name == "pullback02":
                                train_counter_2 = train_counter_2 + 1
                                img_to_folder(image, img_name, train_dir, resize)
                                train_json_list_append(label, train_labels, img_name)
                                train_create_json_file(train_labels, train_dir)
                            
    print("counter: ", counter)
    print("test: ", test_counter)
    print("train 3: ", train_counter_0)
    print("train 2: ", train_counter_1)
    print("train 1: ", train_counter_2)
    print("pos counter: ", positive_counter)
    print("neg counter: ", negative_counter)

def h5py_to_image_csv(test_patients, cv_sets, test_dir, train_dir):
    
    #set values
    counter = 0
    train_counter = 0
    test_counter = 0
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
                                
                                #img_to_folder(image, img_name, test_dir, resize=True)
                                
                                test_df = test_df.append({"image" : img_name}, ignore_index=True)

                                if group[sub_group_name]["labels"][i][0]:
                                    test_df.loc[test_df["image"]== img_name, "target"] = 1
                                else:
                                    test_df.loc[test_df["image"]== img_name, "target"] = 0

                                test_counter += 1


                            if gname in cv_sets[0]:

                                img_to_folder(image, img_name, train_dir, resize=True)

                                train_df = train_df.append({"image" : img_name}, ignore_index=True)
                                train_df.loc[train_df["image"]== img_name, "kfold"] = 0
                                if group[sub_group_name]["labels"][i][0]:
                                    train_df.loc[train_df["image"]== img_name, "target"] = 1
                                else:
                                    train_df.loc[train_df["image"]== img_name, "target"] = 0
                                train_counter = train_counter + 1

                            if gname in cv_sets[1]:

                                img_to_folder(image, img_name, train_dir, resize=True)

                                train_df = train_df.append({"image" : img_name}, ignore_index=True)
                                train_df.loc[train_df["image"]== img_name, "kfold"] = 1
                                if group[sub_group_name]["labels"][i][0]:
                                    train_df.loc[train_df["image"]== img_name, "target"] = 1
                                else:
                                    train_df.loc[train_df["image"]== img_name, "target"] = 0
                                train_counter = train_counter + 1

                            if gname in cv_sets[2]:

                                img_to_folder(image, img_name, train_dir, resize=True)

                                train_df = train_df.append({"image" : img_name}, ignore_index=True)
                                train_df.loc[train_df["image"]== img_name, "kfold"] = 2
                                if group[sub_group_name]["labels"][i][0]:
                                    train_df.loc[train_df["image"]== img_name, "target"] = 1
                                else:
                                    train_df.loc[train_df["image"]== img_name, "target"] = 0
                                train_counter = train_counter + 1
                                
                            counter = counter + 1
    
    print("test counter: ", test_counter)
    print("train counter: ", train_counter)
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


    # Set directory
    filename = r"G:\My Drive\Projektarbeit_ResearchProject\Github\HDF5_files\['kr', 'sb']_anno_anno_var_no1_rl_calcium_nocalcium_cartesian.h5"

    # test patients
    test_patients = ["patient016", "patient019", "patient023", "patient025", "patient026", "patient034", "patient035", "patient044", "patient052", "patient064"]
    
    # cv sets
    cv_sets_100 = [["patient003", "patient005", "patient007", "patient010", "patient012", "patient017", "patient018", "patient024", "patient043", "patient055"] , \
        ["patient002", "patient004", "patient009", "patient011", "patient013", "patient022", "patient032", "patient042", "patient046", "patient060"], \
        ["patient001", "patient014", "patient020", "patient027", "patient036", "patient040", "patient045", "patient048", "patient049", "patient050"]]

    cv_sets_50 = [["patient003", "patient005", "patient007", "patient010"], ["patient002", "patient004", "patient009", "patient011"], \
         ["patient001", "patient014", "patient020", "patient027"]]

    cv_sets_25 = [["patient003", "patient005"], ["patient002", "patient004"], ["patient001", "patient014"]]

    cv_sets_10 = [["patient003"], ["patient002"], ["patient001"]]

    #dir_test = r"\test"
    dir_train = r"G:\My Drive\Projektarbeit_ResearchProject\datasets\IVUS\IVUS_resized\train_val\256\5"

    # h5py_to_image_csv(
    #     test_patients, 
    #     cv_sets=cv_sets_10, 
    #     test_dir=dir_test, 
    #     train_dir=dir_train)

    h5py_to_image_json(
        test_patients=test_patients,
        cv_sets=cv_sets_10,
        resize=True,
        test_dir=None,
        train_dir=dir_train)
