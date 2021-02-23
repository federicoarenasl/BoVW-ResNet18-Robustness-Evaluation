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
from sklearn.svm import SVC
from tqdm import tqdm

# Get dataset of data split
def get_splits(split_n):
    '''

    '''
    path = './data/split_'+str(split_n)+'/split_'+str(split_n)+"_"
    train_df = pd.read_csv(path+'train.csv')
    val_df = pd.read_csv(path+'val.csv')

    return train_df, val_df

# Read and store images
def image_reader(dataframe):
    '''
    '''
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

    image_dict[0] = category_0
    image_dict[1] = category_1

    return image_dict

# Get SIFT features and descriptors
def sift_features(images):
    '''
    Creates descriptors using sift. Takes one parameter that is images dictionary. Return an array whose first 
    index holds the decriptor_list without an order and the second index holds the sift_vectors dictionary which
    holds the descriptors but this is seperated class by class.
    '''
    sift_vectors = {}
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    for key,value in tqdm(images.items()):
        features = []
        for img in tqdm(value):
            kp, des = sift.detectAndCompute(img,None)
            descriptor_list.extend(des) #dico
            features.append(des)
        sift_vectors[key] = features

    return descriptor_list, sift_vectors

# Perform kmeans clustering on descriptors
def kmeans(k, descriptor_list):
    '''
    A k-means clustering algorithm who takes 2 parameter which is number 
    of cluster(k) and the other is descriptors list(unordered 1d array)
    Returns an array that holds central points.
    '''
    kmeans = KMeans(n_clusters = k, n_init=1, verbose=1)
    kmeans.fit(descriptor_list)

    return kmeans

# Extract visual words from descriptors  
def get_histograms(images, k, kmeans):
    '''
    '''    
    hists = []
    classes = []
    sift = cv2.xfeatures2d.SIFT_create()
    for key,value in tqdm(images.items()):
        for img in tqdm(value):
            kp, des = sift.detectAndCompute(img,None)

            hist = np.zeros(k)
            nkp = np.size(kp)

            for d in des:
                index = kmeans.predict([d])
                hist[index] += 1/nkp # Normalization of histograms
        
            hists.append(hist)
            classes.append(key)

    return np.array(hists), np.array(classes)

# Initialize SVM classifier
def train_svm(visual_words, labels):
    # Get training data
    X_train = visual_words
    Y_train = labels

    # Initialize SVM classifier
    svc_classifier = SVC(C=100000000,kernel = "linear")
    svc_classifier.fit(X_train, Y_train)
    
    return svc_classifier


if __name__ == '__main__':
    '''
    # Convert dataframe to dictionary of images
    print('Convert dataframe to dictionary of images')
    split_n = 1
    train_splits, val_splits = get_splits(split_n)

    # Get split 1 data
    print('Get split 1 data')
    train_dict = image_reader(train_splits)
    val_dict = image_reader(val_splits)

    # Get full sift features for training data
    print('Get full sift features for training data...')
    train_descriptor_list, train_sift_vectors = sift_features(train_dict)
    print('Get full sift features for validation data...')
    val_descriptor_list, val_sift_vectors = sift_features(val_dict)
    np.save('output/bovw/train_sift_vectors.npy', train_sift_vectors)

    # Perform kmeans training to get visual words
    print('Perform clustering on training data...')
    train_k_means = kmeans(20, train_descriptor_list)
    print('Perform clustering on validation data...')
    val_k_means = kmeans(20, val_descriptor_list)

    # Get visual words
    print("Get training histograms from kmeans clustering")
    train_histograms, train_classes = get_histograms(train_dict, 20, train_k_means)
    np.save('output/bovw/train_visual_words.npy', train_histograms)
    np.save('output/bovw/train_classes.npy', train_classes)

    print("Get validation histograms from kmeans clustering")
    val_histograms, val_classes = get_histograms(val_dict, 20, val_k_means)
    np.save('output/bovw/val_visual_words.npy', val_histograms)
    np.save('output/bovw/val_classes.npy', val_classes)
    '''
    # Load data
    train_histograms = np.load('output/bovw/train_visual_words.npy')
    train_classes = np.load('output/bovw/train_classes.npy')
    val_histograms = np.load('output/bovw/val_visual_words.npy')
    val_classes = np.load('output/bovw/val_classes.npy')

    # Train SVM classifier
    print("Train SVM classifier")
    svm_classifier = train_svm(train_histograms, train_classes)

    print("Training accuracy:", svm_classifier.score(train_histograms, train_classes))
    print("Validation accuracy:", svm_classifier.score(val_histograms, val_classes))

