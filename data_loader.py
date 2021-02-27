# Library imports
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

# Define data paths
DOGS_PATH = "./catdog/DOGS/"
CATS_PATH = "./catdog/CATS/"
DATA_PATH = "./catdogs.csv"


# Create csv file of all data paths
def create_csv():
    dog_imgs = np.array([["./catdog/DOGS/" + name, 0] for name in os.listdir(DOGS_PATH) if os.path.isfile(os.path.join(DOGS_PATH, name))])
    #dog_imgs = dog_imgs[1:] # Remove the .DS_Store file
    cat_imgs = np.array([["./catdog/CATS/" + name, 1] for name in os.listdir(CATS_PATH) if os.path.isfile(os.path.join(CATS_PATH, name))])
    #cat_imgs = cat_imgs[1:] # Remove the .DS_Store file
    imgs = np.concatenate([dog_imgs, cat_imgs])

    df = pd.DataFrame({"image_id": imgs[:, 0], "abc": np.full(imgs.shape[0], ""), "label": imgs[:, 1]})
    df = df[df.image_id != DOGS_PATH + '.DS_Store'] # Remove the .DS_Store file
    df = df[df.image_id != CATS_PATH + '.DS_Store'] # Remove the .DS_Store file
    
    # Split each breed into 3 folds (A, B and C) with balanced breeds
    classes = [('dog', DOGS_PATH), ('cat', CATS_PATH)]
    breeds = np.arange(1, 13)
    for animal, path in classes:
        for breed in breeds:
            breed_df = df[df.image_id.str.startswith(path + animal + '_{}_'.format(breed))]
            num_images = breed_df.shape[0]
            # Send first third of the breed to the A fold
            for img_id in breed_df[:num_images//3].image_id:
                df.loc[df.image_id == img_id, 'abc'] = 'A'
            # Send second third of the breed to the B fold
            for img_id in breed_df[num_images//3:num_images*2//3].image_id:
                df.loc[df.image_id == img_id, 'abc'] = 'B'
            # Send last third of the breed to the C fold
            for img_id in breed_df[num_images*2//3:].image_id:
                df.loc[df.image_id == img_id, 'abc'] = 'C'
                
    df.to_csv('catdogs.csv', index=False) # Save DataFrame as a csv file

# Create three splits ([train=A+B, test=C], [train=A+C, test=B], [train=B+C, test=A])
def create_splits(dataframe):
    train_folds = [['A', 'B'], ['A', 'C'], ['B', 'C']]
    splits = {}

    # Create splits
    for i, train in enumerate(train_folds):
        split = {}
        split['train'] = dataframe[dataframe.abc.isin(train)]
        split['test'] = dataframe[~dataframe.abc.isin(train)]

        splits['split_{}'.format(i+1)] = split
        
    # Divide each split's training data into train (75%) and validation (25%) with balanced breeds
    classes = [('dog', DOGS_PATH), ('cat', CATS_PATH)]
    breeds = np.arange(1, 13)
    for i in range(1, 4):
        split_train = pd.DataFrame({"image_id": splits['split_{}'.format(i)]['train'].image_id,
                                  "train_val_test": np.full(splits['split_{}'.format(i)]['train'].shape[0], ""),
                                  "label": splits['split_{}'.format(i)]['train'].label})
        for animal, path in classes:
            for breed in breeds:
                breed_df = split_train[split_train.image_id.str.startswith(path + animal + '_{}_'.format(breed))]
                num_images = breed_df.shape[0]
                # 75% training data
                for img_id in breed_df[:num_images*3//4].image_id:
                    split_train.loc[dataframe.image_id == img_id, 'train_val_test'] = 'train'
                # 25% validation data
                for img_id in breed_df[num_images*3//4:].image_id:
                    split_train.loc[dataframe.image_id == img_id, 'train_val_test'] = 'val'
                    
        splits['split_{}'.format(i)]['train'] = split_train
        splits['split_{}'.format(i)]['test'] = pd.DataFrame(
                                    {"image_id": splits['split_{}'.format(i)]['test'].image_id,
                                  "train_val_test": np.full(splits['split_{}'.format(i)]['test'].shape[0], "test"),
                                  "label": splits['split_{}'.format(i)]['test'].label})
        
    return splits


# Create folders to store the new splits in
def create_dirs(root_dir, split_names, classes):
    # Create root directory, if it does not exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    for split_name in split_names:
        # Create split directory, if it does not exist
        if not os.path.exists(root_dir+split_name):
            os.makedirs(root_dir+split_name)
        
        # Create train directory for the current split, if it does not exist
        if not os.path.exists(root_dir+split_name+'/train/'):
            os.makedirs(root_dir+split_name+'/train/')
        # Create validation directory for the current split, if it does not exist
        if not os.path.exists(root_dir+split_name+'/train/'):
            os.makedirs(root_dir+split_name+'/val/')
        # Create test directory for the current split, if it does not exist
        if not os.path.exists(root_dir+split_name+'/test/'):
            os.makedirs(root_dir+split_name+'/test/')

        for c in classes:
            # Create class directories for the current split, if it does not exist
            if not os.path.exists(root_dir+split_name+'/train/'+c+'/'):
                os.makedirs(root_dir+split_name+'/train/'+c+'/')
            if not os.path.exists(root_dir+split_name+'/val/'+c+'/'):
                os.makedirs(root_dir+split_name+'/val/'+c+'/')
            if not os.path.exists(root_dir+split_name+'/test/'+c+'/'):
                os.makedirs(root_dir+split_name+'/test/'+c+'/')

# Copy files from catdog directory to splits directory
def copy_files(file_splits):
    # Initial information definition
    root_dir = './data/'
    split_names = file_splits.keys()
    classes = ['dog', 'cat']
    # Create directories, if necessary
    create_dirs(root_dir, splits, classes)
    
    for split_name in tqdm(split_names):
        # Copy files to train and validation sets
        train_df_dict = {}
        train_files = []
        train_labels = []
        val_df_dict = {}
        val_files = []
        val_labels = []
        test_df_dict = {}
        test_files = []
        test_labels = []

        for row in tqdm(splits[split_name]['train'].itertuples()):
            # row = (index, image_id, train_val_test, label)
            old_path = row[1]
            file_name = row[1].split('/')[-1]
            data_set = row[2]
            class_name = classes[row[3]]
            new_path = root_dir + split_name + '/' + data_set + '/' + classes[row[3]] + '/' + file_name
            #print(old_path, '\t\t', new_path)
            copyfile(old_path, new_path)

            # Store file names and labels
            if data_set == 'train':
                train_files.append(new_path)
                train_labels.append(row[3])
            else:
                val_files.append(new_path)
                val_labels.append(row[3])
        for row in tqdm(splits[split_name]['test'].itertuples()):
            # row = (index, image_id, train_val_test, label)
            old_path = row[1]
            file_name = row[1].split('/')[-1]
            data_set = row[2]
            class_name = classes[row[3]]
            new_path = root_dir + split_name + '/' + data_set + '/' + classes[row[3]] + '/' + file_name
            #print(old_path, '\t\t', new_path)
            copyfile(old_path, new_path)

            # Store file names and labels
            test_files.append(new_path)
            test_labels.append(row[3])

        # Create .csv with training file names and labels
        train_df_dict['image_id'] = train_files
        train_df_dict['label'] = train_labels
        file_name = root_dir+'/'+split_name+'/'+split_name+'_train.csv'
        pd.DataFrame.from_dict(train_df_dict).to_csv(file_name, index=False)

        # Create .csv with validation file names and labels
        val_df_dict['image_id'] = val_files
        val_df_dict['label'] = val_labels
        file_name = root_dir+'/'+split_name+'/'+split_name+'_val.csv'
        pd.DataFrame.from_dict(val_df_dict).to_csv(file_name, index=False)

        # Create .csv with test file names and labels
        test_df_dict['image_id'] = test_files
        test_df_dict['label'] = test_labels
        file_name = root_dir+'/'+split_name+'/'+split_name+'_test.csv'
        pd.DataFrame.from_dict(test_df_dict).to_csv(file_name, index=False)  

# Create data splits
if __name__ == '__main__':
    # Create .csv file with data location
    create_csv()

    # Retrieve data locations dataframe
    print("Loading data...")
    dataset = pd.read_csv('catdogs.csv')
    
    # Create splits
    print("Creating splits...")
    splits = create_splits(dataset)

    # Create directories and copy files to directories
    print("Copying files...")
    copy_files(splits)
