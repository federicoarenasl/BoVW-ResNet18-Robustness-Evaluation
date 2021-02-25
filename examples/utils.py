import numpy as np
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def find_index(image, center):
    count = 0
    ind = 0
    for i in range(len(center)):
        if(i == 0):
           count = distance.euclidean(image, center[i]) 
           #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i]) 
            #dist = L1_dist(image, center[i])
            if(dist < count):
                ind = i
                count = dist
    return ind
    

class Utils:
    
    @staticmethod
    def split_data(self):
        # Split the data for training
    
        # A different approach - just divide the given files into 3 equal (as much as possible) non-overlapping sets, 
        # stratified by breed. Then, just select one for training, one for validation and one for test. Then, rotate the 
        # selection for each fold.
        file_dict = {}
        num_breeds = 12
        num_folds = 3
        
        from pathlib import Path
        import shutil
        root = '../data/catdog'
        import glob
        
        # Iterate through cats and dogs
        for animal, loc in ('cat','../data/catdog/CATS/'),('dog', '../data/catdog/DOGS/'):
            for breed in range(1, num_breeds + 1):           
                # Get the number of files for this breed
                num_breed_files = len(glob.glob(loc + animal + '_' + str(breed) + '_*.png'))
                fold_length = (int)(num_breed_files / 3)
                
                print(animal, breed, num_breed_files, fold_length)
                
                for fold in range(3):
                    start_idx = fold * fold_length
                    
                    # First two fold simply split according to length but the last fold has to take whatever the remainder is
                    if fold < 2:
                        stop_idx = start_idx + fold_length
                    else:
                        stop_idx = num_breed_files
                    
                    dest_dir = root + '/fold' + str(fold) + '/' + animal.upper() + 'S/'
                    print(dest_dir)
                    Path(dest_dir).mkdir(parents=True, exist_ok=True)                
                    
        #            print(start_idx, stop_idx)
                    
                    for idx in range(start_idx, stop_idx):
                        filename = animal + '_' + str(breed) + '_' + str(idx) + '.png'
                        src_path = root + '/' + animal.upper() + 'S/' + filename
                        dest_path = dest_dir + filename
        #                dest_path = root + '/fold' + str(fold) + '/' + animal.upper() + 'S/' + filename
                        
                        shutil.copyfile(src_path, dest_path)
                    
                        print(fold, idx, src_path, dest_path)
                        
    @staticmethod
    def load_images_from_folder(fold):
        images = {}
        folder = "../data/catdog/" + fold
        for filename in os.listdir(folder):
            category = []
            path = folder + "/" + filename
            for cat in os.listdir(path):
                # The zero flag makes it read it in as a greyscale
                img = cv2.imread(path + "/" + cat)
                if img is not None:
                    category.append(img)
            images[filename] = category
        return images
    
    @staticmethod
    def sift_features(images):
        sift_vectors = {}
        descriptor_list = []
        sift = cv2.xfeatures2d.SIFT_create()
        for key,value in images.items():
            features = []
            for img in value:
                kp, des = sift.detectAndCompute(img,None)
               
                
                descriptor_list.extend(des)
                features.append(des)
            sift_vectors[key] = features
        return [descriptor_list, sift_vectors]
    
    @staticmethod
    def dense_sift_features(images):
        sift_vectors = {}
        descriptor_list = []
        sift = cv2.xfeatures2d.SIFT_create()
        for key,value in images.items():
            features = []
            for img in value:
                step = 10
                kps = []
                for i in range(step, len(img), step):
                    for j in range(step, len(img[0]), step):
                        kps.append(cv2.KeyPoint(j, i, step))
                kp, des = sift.compute(img, kps, None)
               
                
                descriptor_list.extend(des)
                features.append(des)
            sift_vectors[key] = features
        return [descriptor_list, sift_vectors]
    
    @staticmethod
    def orb_features(images):
        sift_vectors = {}
        descriptor_list = []
        sift = cv2.ORB_create()
        for key,value in images.items():
            features = []
            for img in value:
                kp, des = sift.detectAndCompute(img,None)
               
                
                descriptor_list.extend(des)
                features.append(des)
            sift_vectors[key] = features
        return [descriptor_list, sift_vectors]
    
    @staticmethod
    def kmeans(k, descriptor_list):
        kmeans = KMeans(n_clusters = k, n_init=10)
        kmeans.fit(descriptor_list)
        visual_words = kmeans.cluster_centers_ 
        return kmeans
    

    @staticmethod
    def image_class(all_bovw, centers):
        dict_feature = {}
        for key,value in all_bovw.items():
            category = []
            for img in value:
                histogram = np.zeros(len(centers))
                for each_feature in img:
                    ind = find_index(each_feature, centers)
                    histogram[ind] += 1
                category.append(histogram)
            dict_feature[key] = category
        return dict_feature
    
    @staticmethod
    def convert(bovw_data):
        X = []
        y = []
        
        for val in bovw_data["CATS"]:
            X.append(val)
            y.append(0)
            
        for val in bovw_data["DOGS"]:
            X.append(val)
            y.append(1)
        
        return X, y
    
