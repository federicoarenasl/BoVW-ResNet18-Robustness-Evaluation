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

DOGS_PATH = "./catdog/DOGS/"
CATS_PATH = "./catdog/CATS/"
DATA_PATH = "./catdogs.csv"

class CatDogDataset(Dataset):
    def __init__(self, 
                 root_dir = "./",
                 image_dir_name = "catdog/",
                 label_csv_name = "catdogs.csv",
                 transform=transforms.ToTensor(),
                 device='cpu'):
        super().__init__()
        self.root_dir = root_dir
        self.image_dir = os.path.join(self.root_dir, image_dir_name)
        self.df_labels = pd.read_csv(os.path.join(self.root_dir, label_csv_name))
        self.transform = transform
        self.device=device
        
    def __getitem__(self, image_id):
        path = os.path.join(self.image_dir, image_id)
        label = np.array(self.df_labels[self.df_labels['image_id'] == image_id]['label'].values[0])


        #print("Label array:", label)

        label = torch.from_numpy(label).to(self.device)
        image = Image.open(path).convert('RGB')
        image = self.transform(image).to(self.device)
        return (image, label)
    
    def __len__(self):
        return len(self.df_labels)
        
class ImageSampler(Sampler):
    def __init__(self, 
                 sample_idx,
                 data_source=DATA_PATH):
        super().__init__(data_source)
        self.sample_idx = sample_idx
        self.df_images = pd.read_csv(data_source)
        
    def __iter__(self):
        image_ids = self.df_images['image_id'].loc[self.sample_idx]
        return iter(image_ids)
    
    def __len__(self):
        return len(self.sample_idx)

class ImageBatchSampler(BatchSampler):
    def __init__(self, 
                 sampler,
                 aug_count=5,
                 batch_size=30,
                 drop_last=True):
        super().__init__(sampler, batch_size, drop_last)
        self.aug_count = aug_count
        assert self.batch_size % self.aug_count == 0, 'Batch size must be an integer multiple of the aug_count.'
        
    def __iter__(self):
        batch = []
        
        for image_id in self.sampler:
            for i in range(self.aug_count):
                batch.append(image_id)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

def create_csv():
    dog_imgs = np.array([["DOGS/" + name, 0] for name in os.listdir(DOGS_PATH) if os.path.isfile(os.path.join(DOGS_PATH, name))])
    dog_imgs = dog_imgs[1:] # Remove the .DS_Store file
    cat_imgs = np.array([["CATS/" + name, 1] for name in os.listdir(CATS_PATH) if os.path.isfile(os.path.join(CATS_PATH, name))])
    cat_imgs = cat_imgs[1:] # Remove the .DS_Store file
    imgs = np.concatenate([dog_imgs, cat_imgs]) 
    df = pd.DataFrame({"image_id": imgs[:, 0], "label": imgs[:, 1]}) 
    df.to_csv('catdogs.csv', index=False) # Save DataFrame as a csv file


def create_split_loaders(dataset, split, aug_count, batch_size):
    train_folds_idx = split[0]
    valid_folds_idx = split[1]
    train_sampler = ImageSampler(train_folds_idx)
    valid_sampler = ImageSampler(valid_folds_idx)
    train_batch_sampler = ImageBatchSampler(train_sampler, 
                                            aug_count, 
                                            batch_size)
    valid_batch_sampler = ImageBatchSampler(valid_sampler, 
                                            aug_count=1, 
                                            batch_size=batch_size,
                                            drop_last=False)
    train_loader = DataLoader(dataset, batch_sampler=train_batch_sampler)
    valid_loader = DataLoader(dataset, batch_sampler=valid_batch_sampler)
    return (train_loader, valid_loader)    

def get_all_split_loaders(dataset, cv_splits, aug_count=5, batch_size=30):
    """Create DataLoaders for each split.

    Keyword arguments:
    dataset -- Dataset to sample from.
    cv_splits -- Array containing indices of samples to 
                 be used in each fold for each split.
    aug_count -- Number of variations for each sample in dataset.
    batch_size -- batch size.
    
    """
    split_samplers = []
    
    for i in range(len(cv_splits)):
        split_samplers.append(
            create_split_loaders(dataset,
                                 cv_splits[i], 
                                 aug_count, 
                                 batch_size)
        )
    return split_samplers


def load_data():
    # Create csv file
    if not path.exists(DATA_PATH):
        create_csv()
    # Retrieve catdog dataset
    df_train = pd.read_csv(DATA_PATH)

    # Get 3-fold splits
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    splits = []

    for train_idx, test_idx in splitter.split(df_train['image_id'], df_train['label']):
        splits.append((train_idx, test_idx))
    
    transform = transforms.Compose([
        # transforms.RandomAffine(degrees=45, translate=(0.05,0.05), scale=(0.8, 1.0)), 
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5), 
        # transforms.CenterCrop((400, 500)), 
        transforms.ToTensor(),
        ])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = CatDogDataset(transform=transform, device=device)

    data_loaders = get_all_split_loaders(dataset, splits, aug_count=1, batch_size=10)

    data_loaders_dict = [{'train': data_loader[0], 'val': data_loader[1]} for data_loader in data_loaders]

    return device, data_loaders_dict

if __name__ == '__main__':
    # Create csv file
    if not path.exists(DATA_PATH):
        create_csv()
    # Retrieve catdog dataset
    df_train = pd.read_csv(DATA_PATH)

    # Get 3-fold splits
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    splits = []

    for train_idx, test_idx in splitter.split(df_train['image_id'], df_train['label']):
        splits.append((train_idx, test_idx))
    
    transform = transforms.Compose([
        # transforms.RandomAffine(degrees=45, translate=(0.05,0.05), scale=(0.8, 1.0)), 
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5), 
        # transforms.CenterCrop((400, 500)), 
        transforms.ToTensor()])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = CatDogDataset(transform=transform, device=device)

    dataloaders = get_all_split_loaders(dataset, splits, aug_count=1, batch_size=1)

    
    for i, (images, labels) in enumerate(dataloaders[0][0]):
        current_image = images.numpy().reshape(224,224,3)
        print(current_image)
        plt.imshow(current_image[0])

    



