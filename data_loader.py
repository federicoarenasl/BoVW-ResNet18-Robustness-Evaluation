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
from PIL import Image
from tqdm import tqdm
from shutil import copyfile

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

def replace_files(dataset):
    root_dir = './data/'

    splits = {1:['A','B','C'], 2:['B','C','A'], 3: ['A', 'C', 'D']}

    for split in splits:
        print(f"On split {split}")
        for line in tqdm(range(len(dataset))):
            old_path = list(dataset['image_id'])[line]
            old_file_name = old_path.split('/')[-1]
            fold = list(dataset['folder'])[line]

            split_name = 'split_'+str(split)
            dog_or_cat = old_file_name.split('_')[0]            

            if fold == splits[split][0] or fold == splits[split][1]:
                new_path = root_dir+split_name+'/train'+'/'+dog_or_cat+'/'+old_file_name
                copyfile(old_path, new_path)
            
            else:
                new_path = root_dir+split_name+'/val'+'/'+dog_or_cat+'/'+old_file_name
                copyfile(old_path, new_path)

if __name__ == '__main__':
    dataset = pd.read_csv('catdogs.csv')
    replace_files(dataset)

