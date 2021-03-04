import numpy as np
from numpy.random import normal, randint
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
from tqdm import tqdm
from shutil import copyfile


class Noises:
    def __init__(self):
        '''
        This class processes the images from a file location
        to output the same, noisier images in an output file location
        '''

    def gaussian_pixel_noise(self, img, std):
        '''
        This function takes an image and a standard deviation as input and outputs
        the same image with noise added to it.
        '''
        gauss = normal(0, std, img.shape).astype('uint8')
        img_gauss = cv2.add(img,gauss)
        image = np.clip(img_gauss, 0, 255)
                        
        return image
    
    def gaussian_blurring(self, image, n_convs):
        '''
        This function takes an image as an input, and the number of convolutions
        to apply to the image, and outputs a blurred image
        '''
        self.convolution_kernel = 1/16 * np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]])
        for conv in range(n_convs):
            image = cv2.filter2D(image, -1, self.convolution_kernel)
        return image
    
    def increase_contrast(self, image, factor):
        '''
        This function takes an image and a factor as an input, and outputs a contrasted
        image according to the given factor
        '''
        n_rows, n_columns, n_channels = image.shape
        for channel in range(n_channels):
            image[:, :, channel] = image[:, :, channel] * factor
        
        image = np.clip(image, 0, 255)

        return image

    def decrease_contrast(self, image, factor):
        '''
        Takes an image and a factor as input, and outputs an image with the contrast decreased
        by the given factor
        '''
        n_rows, n_columns, n_channels = image.shape
        for channel in range(n_channels):
            image[:, :, channel] = image[:, :, channel] * factor
        image = np.clip(image, 0, 255)

        return image
    
    def increase_brightness(self, image, factor):
        '''
        Takes an image and a factor as input, and outputs the same image with the brightness
        increased
        '''
        n_rows, n_columns, n_channels = image.shape
        for channel in range(n_channels):
            image[:, :, channel] += factor
            image = np.clip(image, 0, 255)
        return image
    
    def decrease_brightness(self, image, factor):
        '''
        Takes an image and a factor as input, and outputs the same image with the brightness
        decreased
        '''
        n_rows, n_columns, n_channels = image.shape

        for channel in range(n_channels):
            image[:, :, channel] -= factor
            image = np.clip(image, 0, 255)
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

    def increase_saturation_noise(self,image, std):
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

    def image_occlusion(self,image, edge, print_center=False):
        '''
        Receives an image and an edge and returns the same image 
        with occlusion
        '''
        n_rows, n_columns, _ = image.shape
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
    
class PerturbImages(Noises):
    def __init__(self, directories_created=True, source_path = "./data/full_split_"):
        self.directories_created = directories_created
        self.source_path = source_path # Path for images, add number according to split
        self.robust_path = "./data/robustness"
        self.source_path = source_path

        self.perturb_ids = list(range(1,10))
        self.perturb_levels = list(range(1,11))
        self.splits = list(range(1,4))
        self.classes = ['dog', 'cat']
    
    def create_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def create_placeholder_dirs(self):
        # Create root directory
        self.create_dir(self.robust_path)
        # Create list of sub-directory names

        # Create all directories
        print("Creating directories...")
        for perturb_id in tqdm(self.perturb_ids):
            self.create_dir(self.robust_path)
            for perturb_level in self.perturb_levels:
                for split in self.splits:
                    for class_ in self.classes:
                        # Create perturbation id folder
                        full_path = self.robust_path+'/5_'+str(perturb_id)+'/'+str(perturb_level)+'/full_split_'+str(split)+'/val/'+class_
                        self.create_dir(full_path)

    def perform_perturbations(self):
        # If placeholders have not been created, create placeholders
        if not self.directories_created:
            self.create_placeholder_dirs()
        
        for split in tqdm(self.splits):
            csv_path = self.source_path+str(split)+"/full_split_"+str(split)+"_val.csv"
            split_df = pd.read_csv(csv_path)
            image_paths = list(split_df['image_id'])
            image_ids = list(split_df['label'])
            print(f"Loading {len(image_paths)} images...")
            for i, im_path in tqdm(enumerate(image_paths)):
                # Load current image
                original_image = cv2.imread(im_path)
                image_name = im_path.split('/')[-1]
                curr_class = self.classes[image_ids[i]]

                # Start perturbations of current image
                # Add gaussian noise
                #print(f"Adding noise...")
                #self.add_all_gaussian_noise(original_image, "5_1", split, curr_class, image_name)
                # Add gaussian blurr
                #self.add_all_gaussian_blurr(original_image, "5_2", split, curr_class, image_name)
                #print(f"Done with image {im_path}...")
                # Increase contrast 
                #self.add_all_increase_contrast(original_image, "5_3", split, curr_class, image_name)
                #print(f"Done with image {im_path}...")
                # Decrease contrast 
                # self.add_all_decrease_contrast(original_image, "5_4", split, curr_class, image_name)
                #print(f"Done with image {im_path}...")
                # Increase brightness 
                #self.add_all_increase_brightness(original_image, "5_5", split, curr_class, image_name)
                #print(f"Done with image {im_path}...")
                # Decrease brightness 
                #self.add_all_decrease_brightness(original_image, "5_6", split, curr_class, image_name)
                #print(f"Done with image {im_path}...")
                # Add occlusions
                self.add_all_image_occlusions(original_image, "5_9", split, curr_class, image_name)

    def add_all_gaussian_noise(self, original_image,  id_path, split, curr_class, image_name):
        self.stds = np.arange(0, 19, 2)
        for l, std in enumerate(self.stds):
            image = np.array(original_image).copy()
            image = self.gaussian_pixel_noise(image, std)
            if curr_class == 'dog':
                new_im_path = self.robust_path+"/"+id_path+"/"+str(l+1)+"/full_split_"+str(split)+"/val/dog/"+image_name
                cv2.imwrite(new_im_path,image)
                #print(f"Saving image to {new_im_path}")
                #copyfile(old_path, new_im_path)
            else:
                new_im_path = self.robust_path+"/"+id_path+"/"+str(l+1)+"/full_split_"+str(split)+"/val/cat/"+image_name
                cv2.imwrite(new_im_path,image)
                #print(f"Saving image to {new_im_path}")
                #copyfile(old_path, new_im_path)
    
    def add_all_gaussian_blurr(self, original_image,  id_path, split, curr_class, image_name):
        num_convs = np.arange(10)
        for l, n_convs in enumerate(num_convs):
            image = np.array(original_image).copy() # This allows us to apply the same convolution to the blurred image
            image = self.gaussian_blurring(image, n_convs)
            if curr_class == 'dog':
                new_im_path = self.robust_path+"/"+id_path+"/"+str(l+1)+"/full_split_"+str(split)+"/val/dog/"+image_name
                cv2.imwrite(new_im_path,image)
                print(f"Saving image to {new_im_path}")
                #copyfile(old_path, new_im_path)
            else:
                new_im_path = self.robust_path+"/"+id_path+"/"+str(l+1)+"/full_split_"+str(split)+"/val/cat/"+image_name
                cv2.imwrite(new_im_path,image)
                print(f"Saving image to {new_im_path}")
                #copyfile(old_path, new_im_path)
    
    def add_all_increase_contrast(self, original_image,  id_path, split, curr_class, image_name):
        factors = np.linspace(1.0, 1.27, 10)
        for l,factor in enumerate(factors):
            image = np.array(original_image).copy()
            image = self.increase_contrast(image, factor)
            if curr_class == 'dog':
                new_im_path = self.robust_path+"/"+id_path+"/"+str(l+1)+"/full_split_"+str(split)+"/val/dog/"+image_name
                cv2.imwrite(new_im_path,image)
                print(f"Saving image to {new_im_path}")
                #copyfile(old_path, new_im_path)
            else:
                new_im_path = self.robust_path+"/"+id_path+"/"+str(l+1)+"/full_split_"+str(split)+"/val/cat/"+image_name
                cv2.imwrite(new_im_path,image)
                print(f"Saving image to {new_im_path}")
                #copyfile(old_path, new_im_path)
                
    def add_all_decrease_contrast(self, original_image,  id_path, split, curr_class, image_name):
        factors = np.arange(1.0, 0.0, -0.1)
        for l,factor in enumerate(factors):
            image = np.array(original_image).copy()
            image = self.decrease_contrast(image, factor)
            if curr_class == 'dog':
                new_im_path = self.robust_path+"/"+id_path+"/"+str(l+1)+"/full_split_"+str(split)+"/val/dog/"+image_name
                cv2.imwrite(new_im_path,image)
                print(f"Saving image to {new_im_path}")
                #copyfile(old_path, new_im_path)
            else:
                new_im_path = self.robust_path+"/"+id_path+"/"+str(l+1)+"/full_split_"+str(split)+"/val/cat/"+image_name
                cv2.imwrite(new_im_path,image)
                print(f"Saving image to {new_im_path}")
                #copyfile(old_path, new_im_path)
    
    def add_all_increase_brightness(self, original_image,  id_path, split, curr_class, image_name):
        factors = np.arange(0, 50, 5)
        for l,factor in enumerate(factors):
            image = np.array(original_image).copy()
            image = self.increase_brightness(image, factor)
            if curr_class == 'dog':
                new_im_path = self.robust_path+"/"+id_path+"/"+str(l+1)+"/full_split_"+str(split)+"/val/dog/"+image_name
                cv2.imwrite(new_im_path,image)
                print(f"Saving image to {new_im_path}")
                #copyfile(old_path, new_im_path)
            else:
                new_im_path = self.robust_path+"/"+id_path+"/"+str(l+1)+"/full_split_"+str(split)+"/val/cat/"+image_name
                cv2.imwrite(new_im_path,image)
                print(f"Saving image to {new_im_path}")
                #copyfile(old_path, new_im_path)
    
    def add_all_decrease_brightness(self, original_image,  id_path, split, curr_class, image_name):
        factors = np.arange(0, 50, 5)
        for l,factor in enumerate(factors):
            image = np.array(original_image).copy()
            image = self.decrease_brightness(image, factor)
            if curr_class == 'dog':
                new_im_path = self.robust_path+"/"+id_path+"/"+str(l+1)+"/full_split_"+str(split)+"/val/dog/"+image_name
                cv2.imwrite(new_im_path,image)
                print(f"Saving image to {new_im_path}")
                #copyfile(old_path, new_im_path)
            else:
                new_im_path = self.robust_path+"/"+id_path+"/"+str(l+1)+"/full_split_"+str(split)+"/val/cat/"+image_name
                cv2.imwrite(new_im_path,image)
                print(f"Saving image to {new_im_path}")
                #copyfile(old_path, new_im_path)
    
    def add_all_image_occlusions(self, original_image,  id_path, split, curr_class, image_name):
        edges = np.arange(0, 50, 5)
        for l,edge in enumerate(edges):
            image = np.array(original_image).copy()
            image = self.image_occlusion(image, edge, print_center=True)
            if curr_class == 'dog':
                new_im_path = self.robust_path+"/"+id_path+"/"+str(l+1)+"/full_split_"+str(split)+"/val/dog/"+image_name
                cv2.imwrite(new_im_path,image)
                print(f"Saving image to {new_im_path}")
                #copyfile(old_path, new_im_path)
            else:
                new_im_path = self.robust_path+"/"+id_path+"/"+str(l+1)+"/full_split_"+str(split)+"/val/cat/"+image_name
                cv2.imwrite(new_im_path,image)
                print(f"Saving image to {new_im_path}")
                #copyfile(old_path, new_im_path)

if __name__ == "__main__":
    # Create robustness folder placeholder
    perturbimages = PerturbImages()
    perturbimages.perform_perturbations()
