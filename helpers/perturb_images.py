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
from tqdm.notebook import tqdm


class Noises:
    def __init__(self):
        '''
        This class processes the images from a file location
        to output the same, noisier images in an output file location
        '''
        self.convoltion_kernel = 1/16 * np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]])

    def gaussian_pixel_noise(self, image, std):
        '''
        This function takes an image and a standard deviation as input and outputs
        the same image with noise added to it.
        '''
        n_rows, n_columns, n_channels = image.shape
        #print(f"Standard deviation: {std}")
        for channel in range(n_channels):
            for row in range(n_rows):
                for column in range(n_columns):
                    image[row][column][channel] += normal(loc=0, scale=std)
                    if image[row][column][channel] < 0:
                        image[row][column][channel] = 0
                    elif image[row][column][channel] > 255:
                        image[row][column][channel] = 255
                        
        return image
    
    def gaussian_blurring(self, image, n_convs):
        '''
        This function takes an image as an input, and the number of convolutions
        to apply to the image, and outputs a blurred image
        '''
        for conv in range(n_convs):
            image = cv2.filter2D(image, -1, self.convoltion_kernel)
        return image
    
    def increase_contrast(self, image, factor):
        '''
        This function takes an image and a factor as an input, and outputs a contrasted
        image according to the given factor
        '''
        n_rows, n_columns, n_channels = image.shape

        n_rows, n_columns, n_channels = image.shape
        for channel in range(n_channels):
            image[:, :, channel] = image[:, :, channel] * factor
            for row in range(n_rows):
                for column in range(n_columns):
                    if image[row][column][channel] > 255:
                        image[row][column][channel] = 255
        return image

    def decrease_contrast(self, image, factor):
        '''
        Takes an image and a factor as input, and outputs an image with the contrast decreased
        by the given factor
        '''
        n_rows, n_columns, n_channels = image.shape
        for channel in range(n_channels):
            image[:, :, channel] = image[:, :, channel] * factor
        return image
    
    def increase_brightness(self, image, factor):
        '''
        Takes an image and a factor as input, and outputs the same image with the brightness
        increased
        '''
        n_rows, n_columns, n_channels = image.shape
        for channel in range(n_channels):
            image[:, :, channel] += factor
            for row in range(n_rows):
                for column in range(n_columns):
                    if image[row][column][channel] > 255:
                        image[row][column][channel] = 255
        return image
    
    def decrease_brightness(self, image, factor):
        '''
        Takes an image and a factor as input, and outputs the same image with the brightness
        decreased
        '''
        n_rows, n_columns, n_channels = image.shape

        for channel in range(n_channels):
            image[:, :, channel] -= factor
            for row in range(n_rows):
                for column in range(n_columns):
                    if image[row][column][channel] < 255:
                        image[row][column][channel] = 255
        return image
    
    def increase_hue_noise(self, image, std):
        '''
        Takes an image as input and a standard deviation and returns an image with
        it's hue noise increased
        '''
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        n_rows, n_columns, n_channels = image.shape
        for row in range(n_rows):
            for column in range(n_columns):
                image[row][column][channel] += normal(loc=0, scale=std) * 255
                if image[row][column][channel] < 0:
                    image[row][column][channel] = 0
                elif image[row][column][channel] > 255:
                    image[row][column][channel] -= 255
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        
        return image

    def increase_saturation_noise(image, std):
        '''
        Receives an image as input and outputs the same image with increased 
        saturation
        '''
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        n_rows, n_columns, n_channels = image.shape
        for row in range(n_rows):
            for column in range(n_columns):
                image[row][column][channel] += normal(loc=0, scale=std) * 255
                if image[row][column][channel] < 0:
                    image[row][column][channel] = 0
                elif image[row][column][channel] > 255:
                    image[row][column][channel] = 255
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        return image

    def image_occlusion(image, edge, print_center=False):
        '''
        Receives an image and an edge and returns the same image 
        with occlusion
        '''
        _rows, n_columns, _ = image.shape
        center = np.array([randint(0, n_rows+1), randint(0, n_columns+1)])
        start_point = center - edge // 2
        start_point = tuple((start_point[0], start_point[1]))
        end_point = center + edge // 2
        end_point = tuple((end_point[0], end_point[1]))
        colour = (0, 0, 0) 
        thickness = -1
        if print_center:
            print(f"Center of the square: [{center[0]}, {center[1]}]")
        image = cv2.rectangle(image, start_point, end_point, colour, thickness)

        return image
    
class PerturbImages:
    def __init__(self, source_path="./data/full_split_"):
        self.source_path = source_path # Path for images, add number according to split
        self.robust_path = "./data/robustness"
    
    def create_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def create_placeholder_dirs(self,):
        # Create root directory
        self.create_dir(self.robust_path)
        perturb_ids = list(range(1,10))
        perturb_levels = list(range(1,11))
        splits = list(range(1,4))
        classes = ['cat', 'dog']
        for perturb_id in tqdm(perturb_ids):
            for perturb_level in perturb_levels:
                for split in splits:
                    for class_ in classes:
                        # Create perturbation id folder
                        full_path = '/5_'+str(perturb_id)+'/'+str(perturb_level)+'/full_split_'+str(split)+'/val/'+class_
                        self.create_dir(full_path)


if __name__ == "__main__":
    # Create robustness folder placeholder
