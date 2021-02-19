import os.path
from os import path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
from tqdm import tqdm
from shutil import copyfile
from sklearn.utils import shuffle

DOGS_PATH = "./catdog/DOGS/"
CATS_PATH = "./catdog/CATS/"
DATA_PATH = "./catdogs.csv"


def create_csv():
    dog_imgs = np.array([["./catdog/DOGS/" + name, 0] for name in os.listdir(DOGS_PATH) if os.path.isfile(os.path.join(DOGS_PATH, name))])
    dog_imgs = dog_imgs[1:] # Remove the .DS_Store file
    cat_imgs = np.array([["./catdog/CATS/" + name, 1] for name in os.listdir(CATS_PATH) if os.path.isfile(os.path.join(CATS_PATH, name))])
    cat_imgs = cat_imgs[1:] # Remove the .DS_Store file
    imgs = np.concatenate([dog_imgs, cat_imgs]) 

    a = np.full(imgs.shape[0]//3, 'A')
    b = np.full(imgs.shape[0]//3, 'B')
    c = np.full(imgs.shape[0]//3, 'C')
    folders = np.concatenate([a, b, c])
    np.random.shuffle(folders)

    df = pd.DataFrame({"image_id": imgs[:, 0], "folder": folders, "label": imgs[:, 1]}) 
    df.to_csv('catdogs.csv', index=False) # Save DataFrame as a csv file


def create_dirs(root_dir, splits, classes):
    # Create root directory, if it does not exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    for split in splits:
        split_name = 'split_'+str(split)
        # Create split directory, if it does not exist
        if not os.path.exists(root_dir+split_name):
            os.makedirs(root_dir+split_name)
        
        # Create train directory for the current split, if it does not exist
        if not os.path.exists(root_dir+split_name+'/train/'):
            os.makedirs(root_dir+split_name+'/train/')
        # Create validation directory for the current split, if it does not exist
        if not os.path.exists(root_dir+split_name+'/train/'):
            os.makedirs(root_dir+split_name+'/train/')

        for c in classes:
            # Create class directories for the current split, if it does not exist
            if not os.path.exists(root_dir+split_name+'/train/'+c+'/'):
                os.makedirs(root_dir+split_name+'/train/'+c+'/')
            if not os.path.exists(root_dir+split_name+'/val/'+c+'/'):
                os.makedirs(root_dir+split_name+'/val/'+c+'/')

def create_split_indexes(data):
    X_indexes = np.array(list(data.index))
    Y = np.array(list(data['label']))

    kfold = StratifiedShuffleSplit(n_splits=3, test_size=1/3, random_state=0)
    kfold.get_n_splits(X_indexes, Y)

    splits = {}
    split_counter = 0
    for train_index, test_index in kfold.split(X_indexes, Y):
        split_counter += 1
        train_val = {}
        X_train, X_test = X_indexes[train_index], X_indexes[test_index]
        train_val['train'] = X_train
        train_val['val'] = X_test
        splits[split_counter] = train_val
    
    return splits

def map_to_file(splits, data):
    file_names = list(data['image_id'])
    for split, sets in splits.items():
        for set_, indexes in sets.items():
            paths = []
            for i in indexes:
                paths.append(file_names[i])
            splits[split][set_] = paths
    return splits

def copy_files(file_splits):
    root_dir = './data/'
    splits = [1,2,3]
    classes = ['cat', 'dog']

    # Create directories, if necessary
    create_dirs(root_dir, splits, classes)
    print("Going through each split")
    for split, sets in tqdm(file_splits.items()):
        for set_, file_names in sets.items():
            print(f"Copying {set_} files")
            for file_name in tqdm(file_names):
                # Get old file name
                old_path = file_name
                old_file_name = old_path.split('/')[-1]
                # Get new file name
                split_name = 'split_'+str(split)
                dog_or_cat = old_file_name.split('_')[0]
                new_path = root_dir+split_name+'/'+set_+'/'+dog_or_cat+'/'+old_file_name
                # Copy data to new folders
                copyfile(old_path, new_path)


if __name__ == '__main__':
    print("Loading data...")
    # Retreiving data locations dataframe
    dataset = pd.read_csv('catdogs.csv')
    # Create split indexes
    splits_dict = create_split_indexes(dataset)
    # Map indices to file_names
    file_splits = map_to_file(splits_dict, dataset)
    # Create directories and copy files to directories
    copy_files(file_splits)
