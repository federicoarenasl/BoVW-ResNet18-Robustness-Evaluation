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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from tqdm import tqdm

# Define BovW class
class BovW:
    def __init__(self, split_n, data_dir, full_split = False):
        self.split_n = split_n
        self.full_split = full_split
        if self.full_split:
            self.data_dir = data_dir+"/full_split_"
        self.data_dir = data_dir+"/split_"

    def get_splits(self, split_n):
        '''
        Receives the number of the split split_n and outputs the training and validation
        pandas dataframes.
        '''
        if self.full_split:
            path = self.data_dir+str(split_n)+'/full_split_'+str(split_n)+"_"
        else:
            path = self.data_dir+str(split_n)+'/split_'+str(split_n)+"_"
        
        train_df = pd.read_csv(path+'train.csv')
        val_df = pd.read_csv(path+'val.csv')

        return train_df, val_df

    def image_reader(self, dataframe):
        '''
        Receives the dataframe that points to the images, and outputs a dictionnary 
        with all images loaded with OpenCV.
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
    
    def sift_features(self, images):
        '''
        Creates descriptors using sift. Takes one parameter that is images dictionary. Return an array whose first 
        index holds the decriptor_list without an order and the second index holds the sift_vectors dictionary which
        holds the descriptors but this is seperated class by class.
        '''
        sift_vectors = {}
        descriptor_list = []
        sift = cv2.xfeatures2d.SIFT_create()
        for key,value in images.items():
            features = []
            for img in tqdm(value):
                kp, des = sift.detectAndCompute(img,None)
                descriptor_list.extend(des)
                features.append(des)
            sift_vectors[key] = features

        return descriptor_list, sift_vectors
    
    def kmeans(self, k, descriptor_list):
        '''
        A k-means clustering algorithm who takes 2 parameter which is number 
        of cluster(k) and the other is descriptors list(unordered 1d array)
        Returns an array that holds central points.
        '''
        kmeans = KMeans(n_clusters = k, n_init=10, verbose=0)
        kmeans.fit(descriptor_list)
        vwords = kmeans.cluster_centers_

        return kmeans, vwords

    def find_index(self, image, center):
        '''
        Receives an image and a cluster center and returns to which image does the center belong
        as and index
        '''
        count = 0
        ind = 0
        for i in range(len(center)):
            if(i == 0):
                count = distance.euclidean(image, center[i]) 
            else:
                dist = distance.euclidean(image, center[i]) 
                if(dist < count):
                    ind = i
                    count = dist
        return ind

    # Extract visual words from descriptors  
    def get_histograms_dictio(self, sift_vectors, kmeans_centers):
        '''
        Receives the keypoints and the cluster centers and returns 
        a dictionary with all the histograms, divided by class
        '''    
        dict_feature = {}
        for key,value in sift_vectors.items():
            print(f"Getting histograms of class {key}...")
            category = []
            for img in tqdm(value):
                histogram = np.zeros(len(kmeans_centers))
                for each_feature in img:
                    ind = self.find_index(each_feature, kmeans_centers)
                    histogram[ind] += 1
                category.append(histogram)
            dict_feature[key] = category
        return dict_feature

    # Map dictionary of histograms to np arrays
    def get_histogram_arrays(self, sift_vectors, kmeans_centers):
        '''
        Receives the keypoints and cluster centers and returns the np.array of the
        training data and the training labels
        '''
        histogram_dictio = self.get_histograms_dictio(sift_vectors, kmeans_centers)
        X = []
        Y = []
        print("Converting to np.arrays...")
        for key in histogram_dictio.keys():
            for value in histogram_dictio[key]:
                X.append(value)
                Y.append(key)
        
        return np.array(X), np.array(Y)

    def get_all_histograms(self, K_clusters=20):
        '''
        Receives the number of clusters to perform kmeans on the data, and returns nothing
        but outputs a .npy file with the training and validation histograms for the current
        data split
        '''
        # Get split data
        print('Convert dataframe to dictionary of images')
        split_n = self.split_n
        train_splits, val_splits = self.get_splits(split_n)

        # Get split images
        print('Get split 1 data')
        train_dict = self.image_reader(train_splits)
        val_dict = self.image_reader(val_splits)

        # Get descriptors and keypoints from split
        print('Get full sift features for training data')
        train_descriptor_list, train_sift_vectors = self.sift_features(train_dict)
        print('Get full sift features for validation data...')
        val_descriptor_list, val_sift_vectors = self.sift_features(val_dict)

        # Perform kmeans training to get visual words
        K = K_clusters
        print('Perform clustering on training data...')
        train_k_means, train_centers = self.kmeans(K_clusters, train_descriptor_list)
        print('Perform clustering on validation data...')
        val_k_means, valid_centers = self.kmeans(K_clusters, val_descriptor_list)

        # Get histograms and save them
        print("Get histograms from kmeans clustering")
        train_histograms, train_classes = self.get_histogram_arrays(train_sift_vectors, train_centers)
        np.save("output/bovw/split_"+str(self.split_n)+"/histograms/train_visual_words_k_"+str(K_clusters)+".npy", train_histograms)
        np.save("output/bovw/split_"+str(self.split_n)+"/histograms/train_classes_k_"+str(K_clusters)+".npy", train_classes)
        print("Get validation histograms from kmeans clustering")
        val_histograms, val_classes = self.get_histogram_arrays(val_sift_vectors, valid_centers)
        np.save("output/bovw/split_"+str(self.split_n)+"/histograms/val_visual_words_k_"+str(K_clusters)+".npy", val_histograms)
        np.save("output/bovw/split_"+str(self.split_n)+"/histograms/val_classes_k_"+str(K_clusters)+".npy", val_classes)



