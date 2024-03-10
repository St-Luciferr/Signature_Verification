import os
import shutil

import random

def data_split(dataset_dir, outut_dir, split_ratio=0.7):
    sub_dirs = os.listdir(dataset_dir)
    train_len=int(len(sub_dirs)*split_ratio)
    test_len=int(len(sub_dirs)*(1-split_ratio)/2)
    val_len=len(sub_dirs)-train_len-test_len
    train_dir=os.path.join(outut_dir,"train")
    test_dir=os.path.join(outut_dir,"test")
    val_dir=os.path.join(outut_dir,"val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_dirs=random.sample(sub_dirs,train_len)
    test_dirs=random.sample(sub_dirs, test_len)
    val_dirs=random.sample(sub_dirs, val_len)

    for dirs in train_dirs:
        shutil.copytree(os.path.join(dataset_dir,dirs),os.path.join(train_dir,dirs))
    for dirs in test_dirs:
        shutil.copytree(os.path.join(dataset_dir,dirs),os.path.join(test_dir,dirs))
    for dirs in val_dirs:
        shutil.copytree(os.path.join(dataset_dir,dirs),os.path.join(val_dir,dirs))

if __name__ == "__main__":
    dataset_directory = r"E:\QuickFox\Signature\signature_dataset\CEDAR"
    output_directory = r"E:\QuickFox\Signature\signature_dataset\split"
    data_split(dataset_directory, output_directory)
