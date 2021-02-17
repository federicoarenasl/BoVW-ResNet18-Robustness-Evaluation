# Library import
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from tqdm import tqdm

def csv_to_splits_df(filename):
    df = pd.read_csv(filename)

    splits_train = []
    splits_val = []

    divs = ['C', 'B', 'A']

    for div in divs:
        # Create training set for current split
        train_df = shuffle(df.loc[df.folder != div], random_state=0)
        splits_train.append(train_df[['image_id', 'label']]) # folder column is removed
        # Create validation set for current split
        val_df = shuffle(df.loc[df.folder == div], random_state=0)
        splits_val.append(val_df[['image_id', 'label']]) # folder column is removed

    return splits_train, splits_val


def image_reader(dataframe):
    image_dict = {}
    file_locations = list(dataframe['image_id'])
    labels = list(dataframe['label'])
    category_0 = []
    category_1 = []
    for i in range(len(file_locations)):
        image = cv2.imread(file_locations[i], cv2.COLOR_RGB2BGR)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except: # if the image is gray
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        
        if labels[i] == 0:
            category_0.append(image)
        else:
            category_1.append(image)

    image_dict['0'] = category_0
    image_dict['1'] = category_1

    return image_dict


# Creates descriptors using sift 
# Takes one parameter that is images dictionary
# Return an array whose first index holds the decriptor_list without an order
# And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
def sift_features(images):
    sift_vectors = {}
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    for key,value in tqdm(images.items()):
        features = []
        for img in tqdm(value):
            kp, des = sift.detectAndCompute(img,None)
           
            descriptor_list.extend(des)
            features.append(des)
        sift_vectors[key] = features
    return [descriptor_list, sift_vectors]

# A k-means clustering algorithm who takes 2 parameter which is number 
# of cluster(k) and the other is descriptors list(unordered 1d array)
# Returns an array that holds central points.
def kmeans(k, descriptor_list):
    kmeans = KMeans(n_clusters = k, n_init=1, verbose=1)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_ 
    labels = kmeans.labels_
    return visual_words, labels


# Convert dataframe to dictionary of images
print('Convert dataframe to dictionary of images')
train_splits, val_splits = csv_to_splits_df("catdogs.csv")

# Get split 1 data
print('Get split 1 data')
split_index = 0
train_dict = image_reader(train_splits[split_index])
val_dict = image_reader(val_splits[split_index])

# Get full sift features for training data
print('Get full sift features for training data')
sifts = sift_features(train_dict) 
# Takes the descriptor list which is unordered one
descriptor_list = sifts[0] 
# Takes the sift features that is seperated class by class for train data
all_bovw_feature = sifts[1] 
# Takes the sift features that is seperated class by class for test data
test_bovw_feature = sift_features(val_dict)[1]

# Takes the central points which is visual words
print('Takes the central points which is visual words')
visual_words, predictions, labels = kmeans(150, descriptor_list) 
# np.save('visual_words.npy', visual_words)
# np.save('labels.npy', labels)
visual_words = np.load('visual_words.npy')
labels = np.load('labels.npy')
