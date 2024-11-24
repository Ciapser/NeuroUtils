import cv2 
import os
import numpy as np
from timeit import default_timer as timer   
import skimage as sk
import tensorflow as tf
from tqdm import tqdm
import random
from scipy.ndimage import rotate
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import itertools
from PIL import Image
from matplotlib.widgets import Slider
import hashlib
import math
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)
import json
import inspect
import numba as nb
from texttable import Texttable

class DataSets:
    #Scenario 1
    #If it is classification data
    ################################################
    @nb.njit(parallel=True)
    def Min_max_scaling(Data, Data_minimum = None, Data_maximum = None, New_min = 0, New_max = 1, DataType = np.float32):
        #For training data where its not necessary to specify it
        # For training data where it's not necessary to specify the min/max
        if Data_minimum is None:
            print("Didnt provided data minimum --> It will be calculated from provided Data\n !!This may lead to DATA LEAKAGE if not carefully watched!!")
            Data_minimum = np.empty(1, dtype=DataType)
            for c in range(1):
                Data_minimum[c] = DataType(Data.min())
               
        if Data_maximum is None:
            print("Didnt provided data maximum --> It will be calculated from provided Data\n !!This may lead to DATA LEAKAGE if not carefully watched!!")
            Data_maximum = np.empty(1, dtype=DataType)
            for c in range(1):
                Data_maximum[c] = DataType(Data.max())  
            
        # Create an empty array for the transformed data
        Normalized_data = np.empty_like(Data, dtype = DataType)
        
        # Process the array in chunks to save memory
        for i in nb.prange(Data.shape[0]):
            Normalized_data[i] = ((Data[i] - Data_minimum[0]) / (Data_maximum[0] - Data_minimum[0])) * (New_max - New_min) + New_min  
        
        return Normalized_data
    
    
    
    @nb.njit(parallel=True)
    def Min_max_scaling_channel_wise(Data, Channel_data_minimum=None, Channel_data_maximum=None, New_min=0, New_max=1, DataType=np.float32):
        # Data shape is assumed to be (n, img_h, img_w, channels)
        n, img_h, img_w, channels = Data.shape
        
        # For training data where it's not necessary to specify the min/max
        if Channel_data_minimum is None:
            print("Didnt provided Channel data minimum --> It will be calculated from provided Data\n !!This may lead to DATA LEAKAGE if not carefully watched!!")
            Channel_data_minimum = np.empty(channels, dtype=DataType)
            for c in range(channels):
                Channel_data_minimum[c] = DataType(Data[:, :, :, c].min())
        
        if Channel_data_maximum is None:
            print("Didnt provided Channel data maximum --> It will be calculated from provided Data\n !!This may lead to DATA LEAKAGE if not carefully watched!!")
            Channel_data_maximum = np.empty(channels, dtype=DataType)
            for c in range(channels):
                Channel_data_maximum[c] = DataType(Data[:, :, :, c].max())
                
        # Create an empty array for the transformed data
        Normalized_data = np.empty_like(Data, dtype=DataType)
        
        # Process the array in chunks to save memory, applying scaling channel-wise
        for i in nb.prange(n):
            for c in range(channels):
                Normalized_data[i, :, :, c] = ((Data[i, :, :, c] - Channel_data_minimum[c]) / 
                                               (Channel_data_maximum[c] - Channel_data_minimum[c])) * (New_max - New_min) + New_min
        
        return Normalized_data
    
    ###################################################################
    @nb.njit(parallel=True)
    def Z_score_norm(Data, Mean = None, Std = None, DataType = np.float32):
        #For training data where its not necessary to specify it
        if Mean is None:
            print("Didnt provided mean --> It will be calculated from provided Data\n !!This may lead to DATA LEAKAGE if not carefully watched!!")
            Mean = Data.mean()
        if Std is None:
            print("Didnt provided std --> It will be calculated from provided Data\n !!This may lead to DATA LEAKAGE if not carefully watched!!")
            Std = Data.std()
        # Create an empty array for the transformed data
        Normalized_data = np.empty_like(Data, dtype = DataType)
        
        # Process the array in chunks to save memory
        for i in nb.prange(Data.shape[0]):
            Normalized_data[i] = (Data[i] - Mean) / Std
        
        return Normalized_data
        
    
    @nb.njit(parallel=True)
    def Z_score_norm_channel_wise(Data, Channel_mean=None, Channel_std=None, DataType=np.float32):
        # Data shape is assumed to be (n, img_h, img_w, channels)
        n, img_h, img_w, channels = Data.shape
        
        # For training data where it's not necessary to specify the mean/std
        if Channel_mean is None:
            print("Didnt provided Channel mean --> It will be calculated from provided Data\n !!This may lead to DATA LEAKAGE if not carefully watched!!")
            Channel_mean = np.empty(channels, dtype=DataType)
            for c in range(channels):
                Channel_mean[c] = Data[:, :, :, c].mean()
    
        if Channel_std is None:
            print("Didnt provided Channel std --> It will be calculated from provided Data\n !!This may lead to DATA LEAKAGE if not carefully watched!!")
            Channel_std = np.empty(channels, dtype=DataType)
            for c in range(channels):
                Channel_std[c] = Data[:, :, :, c].std()
                
        # Create an empty array for the transformed data
        Normalized_data = np.empty_like(Data, dtype=DataType)
        
        # Process the array in chunks to save memory, applying normalization channel-wise
        for i in nb.prange(n):
            for c in range(channels):
                Normalized_data[i, :, :, c] = (Data[i, :, :, c] - Channel_mean[c]) / Channel_std[c]
        
        return Normalized_data
    
    
    ##########################################################################
    @nb.njit(parallel=True)
    def Max_absolute_scaling(Data, Data_maximum = None, DataType = np.float32):
        #For training data where its not necessary to specify it
        if Data_maximum is None:
            print("Didnt provided Data maximum --> It will be calculated from provided Data\n !!This may lead to DATA LEAKAGE if not carefully watched!!")
            Data_maximum = np.empty(1, dtype=DataType)
            for c in range(1):
                Data_maximum[c] = DataType(Data.max())  
                
        # Create an empty array for the transformed data
        Normalized_data = np.empty_like(Data, dtype = DataType)
        # Process the array in chunks to save memory
        for i in nb.prange(Data.shape[0]):
            Normalized_data[i] = Data[i] / Data_maximum[0]
        
        return Normalized_data
    
    
    @nb.njit(parallel=True)
    def Max_absolute_scaling_channel_wise(Data, Channel_data_maximum = None, DataType=np.float32):
        # Data shape is assumed to be (n, img_h, img_w, channels)
        n, img_h, img_w, channels = Data.shape
        
        # For training data where it's not necessary to specify the maximum value
        if Channel_data_maximum is None:
            print("Didnt provided Channel data maximum --> It will be calculated from provided Data\n !!This may lead to DATA LEAKAGE if not carefully watched!!")
            Channel_data_maximum = np.empty(channels, dtype=DataType)
            for c in range(channels):
                Channel_data_maximum[c] = np.max(np.abs(Data[:,:,:,c]))
        
        # Create an empty array for the transformed data
        Normalized_data = np.empty_like(Data, dtype=DataType)
        
        # Process the array in chunks to save memory, applying scaling channel-wise
        for i in nb.prange(n):
            for c in range(channels):
                Normalized_data[i, :, :, c] = Data[i, :, :, c] / Channel_data_maximum[c]
        
        return Normalized_data

    
    def Robust_scaling(Data, Median=None, IQR=None, DataType=np.float32):
        # Calculate Median and IQR if not provided (done outside numba)
        if Median is None:
            print("Didnt provided median --> It will be calculated from provided Data\n !!This may lead to DATA LEAKAGE if not carefully watched!!")
            Median = np.median(Data)
        if IQR is None:
            print("Didnt provided IQR --> It will be calculated from provided Data\n !!This may lead to DATA LEAKAGE if not carefully watched!!")
            q1 = np.percentile(Data, 25)
            q3 = np.percentile(Data, 75)
            IQR = q3 - q1
    
        # Create an empty array for the transformed data
        Normalized_data = np.empty(Data.shape, dtype=DataType)
    
        # Define the numba function for the scaling process
        @nb.njit(parallel=True)
        def _robust_scaling_numba(Data, Median, IQR, Normalized_data):
            for i in nb.prange(Data.shape[0]):
                Normalized_data[i] = (Data[i] - Median) / IQR
    
        # Apply the numba-optimized scaling
        _robust_scaling_numba(Data, Median, IQR, Normalized_data)
    
        return Normalized_data

    
    def Robust_scaling_channel_wise(Data, Channel_median=None, Channel_IQR=None, DataType=np.float32):
        # Data shape is assumed to be (n, img_h, img_w, channels)
        _, _, _, channels = Data.shape
    
        # Calculate Median and IQR for each channel if not provided (done outside numba)
        if Channel_median is None:
            print("Didnt provided Channel median --> It will be calculated from provided Data\n !!This may lead to DATA LEAKAGE if not carefully watched!!")
            Channel_median = np.empty(channels, dtype=DataType)
            for c in range(channels):
                Channel_median[c] = np.median(Data[:, :, :, c])
    
        if Channel_IQR is None:
            print("Didnt provided Channel IQR --> It will be calculated from provided Data\n !!This may lead to DATA LEAKAGE if not carefully watched!!")
            Channel_IQR = np.empty(channels, dtype=DataType)
            for c in range(channels):
                q1 = np.percentile(Data[:, :, :, c], 25)
                q3 = np.percentile(Data[:, :, :, c], 75)
                Channel_IQR[c] = q3 - q1
    
        # Create an empty array for the transformed data
        Normalized_data = np.empty(Data.shape, dtype=DataType)
    
        # Define the numba function for the scaling process
        @nb.njit(parallel=True)
        def _robust_scaling_channel_wise_numba(Data, Channel_median, Channel_IQR, Normalized_data):
            n, _, _, channels = Data.shape
            for i in nb.prange(n):
                for c in range(channels):
                    Normalized_data[i, :, :, c] = (Data[i, :, :, c] - Channel_median[c]) / Channel_IQR[c]
    
        # Apply the numba-optimized scaling
        _robust_scaling_channel_wise_numba(Data, Channel_median, Channel_IQR, Normalized_data)
    
        return Normalized_data

    
    
    def Reduce_Img_Classification_Class_Size(X,samples_per_class,shuffle_img = False):
        if shuffle_img:
            np.random.shuffle(X)
        X = X[0:samples_per_class].copy()
  
        if samples_per_class>len(X):
            print("Not enough images in class.   Loaded",len(X),"/",samples_per_class,"images")
        return X
    

    
    
    def Create_Img_Classification_DataSet(Data_directory , img_H , img_W ,Save_directory, grayscale = False , r_value = 1.4 , reduced = False):

        
        
        timer_start = timer()
        
        Classes_list  = os.listdir(Data_directory)

        #creating X sets
        for class_name in Classes_list:
            DataSet=[]
            try:
                Class_directory = os.path.join(Data_directory , class_name)
                InClass_images = os.listdir(Class_directory)
                Progress_desc = "Creating " + class_name + " class"

            except Exception:
                print("Found unsupported format, try to extract images into folder. File:  ", class_name) 
                continue
            
            for image in tqdm(InClass_images, desc = Progress_desc):
                
                #Load Image
                try:

                    img_dir = os.path.join(Class_directory , image)
                    img = cv2.imread(img_dir)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                    img = np.array(img , dtype = np.uint8)
                    
                    if grayscale:
                        if img.shape[2] == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        elif img.shape[2] ==1:
                            pass
                        else:
                            print("Image channels not appropriate: skipping...")
                            continue
                        
                    if not grayscale:
                        if img.shape[2] == 3:
                            pass
                        elif img.shape[2] ==1:
                            print("Image in grayscale while operating on RGB: skipping....")
                        else:
                            print("Image channels not appropriate: skipping...")
                            continue
                            
            
                    
                    
            
                    
                    h = np.shape(img)[0]
                    w = np.shape(img)[1]
                
                    if w/h >= 1/r_value and w/h <=r_value:
                        img = cv2.resize(img, (img_H, img_W), interpolation = cv2.INTER_LANCZOS4)
                        
                        
                    #########
                    #Random Crop
                    #Calculate proportions to keep aspect ratio of image
                    if(img.shape[0] < img.shape[1]):
            
                        height_percentage = (img_H/img.shape[0])
                        width_size = round(img.shape[1]*height_percentage)
                        height_size = img_H
                    elif(img.shape[0] > img.shape[1]):
                        width_percentage = (img_W/img.shape[1])
                        width_size = img_W
                        height_size = round(img.shape[0]*width_percentage)
                    else:
                        width_size = img_W
                        height_size = img_H
                        
                        
                    
                    #Take random crop of the image with preserved resolution and aspect ratio (part of image is just cutted off)
                    img = sk.transform.resize(img,(height_size,width_size),mode='constant',preserve_range=True)
                    
                    if grayscale:
                        img = tf.image.random_crop(value = img ,size=(img_H,img_W))
                        img = np.array([img]).reshape(img_H,img_W,1)
                    if not grayscale:
                        img = tf.image.random_crop(value = img ,size=(img_H,img_W,3))
                        img = np.array([img]).reshape(img_H,img_W,3)
                      
                    DataSet.append(   img   )
            
            
                        
            
            
                    
                #If file is damaged or broken then it occurs
                except Exception as e:
                    #Print Exception
                    print("\nCould not load file: ", image)
                    print(e)
                    

        

            
            DataSet = np.array(DataSet , dtype = np.uint8)
            
            class_save_dir = str(str(class_name)+".npy")
            np.save(os.path.join(Save_directory , class_save_dir), DataSet)
            
            #Making list of classes which are done #2
            print("Done")
            print("----------------------------------------------")
        
        

              
        
        
        print("----------------------\nTime took compiling images:",round(timer()-timer_start,2),"\n----------------------")


    def Create_Img_Classification_DataSet_CSV(Database_directory , img_H , img_W, Save_directory):
        Set_list  = os.listdir(Database_directory)

        for Set in Set_list:
            try:
                directory = os.path.join(Database_directory , str(Set))
                data = pd.read_csv(directory)
                #Save dataframe directly into data folder if its sample submission
                if str(Set) == "sample_submission.csv":
                    data.to_csv(os.path.join(Save_directory , Set),index =False)
                    continue
                    
                #Checking if label is present in data, it indicates its not test data
                if 'label' in data.columns:
                    labels = np.array(data["label"].values , dtype = int)
                    n_classes = max(labels)+1
                    y = General.OneHot_decode(labels,n_classes)
                    data.drop(columns=['label'], inplace=True)
                
                else:
                    y = None    
                lenght = len(data.iloc[0,:].values)
                
                if img_H*img_W == lenght:
                    
                    x = []
                    description = str("Preparing "+Set)
                    for k in tqdm(range(len(data)) , desc = description):
                        img = np.zeros((img_H,img_W),dtype = np.uint8)
                        for i in range(img_H):
                            img[i,:] = data.iloc[k,i*img_W:(i+1)*img_W]
                        x.append(img)
                    x = np.array(x)
                    
                    
                    
                    
                    x_set_save_dir = str("x_"+str(Set.replace(".csv",""))+".npy")
                    y_set_save_dir = str("y_"+str(Set.replace(".csv",""))+".npy")
                    
                    np.save(os.path.join(Save_directory , x_set_save_dir), x)
                    np.save(os.path.join(Save_directory , y_set_save_dir), y)
                    
                    #Making list of sets which are done #2
                    print("Done")
                    print("----------------------------------------------")
                    
                    
                    
                else:
                    print("Mismatch in provided image resolution and expected one. By default if square it should be:",img_H,"x",img_W)
            
            except Exception as e:
                #Print Exception
                print("\nCould not load file: ", Set)
                print(e)
          
       
        
            
    def Load_And_Merge_DataSet(Data_directory, samples_per_class = None, reduced_class_shuffle = False ):

        Classes_list  = os.listdir(Data_directory)
        n_classes = len(Classes_list)
        ClassSet=np.zeros((0,n_classes) , dtype = np.uint8)
        Dictionary = []
        try:
            temporary_sheet = np.load(os.path.join(Data_directory , Classes_list[0]) , mmap_mode='r')
        except:
            try:
                temporary_sheet = np.load(os.path.join(Data_directory , Classes_list[0]) , mmap_mode='r',allow_pickle = True)
            except:
                print("Unsuported file in folde, pergaps some sample submission? If you already have dataset set to x_train,x_val, skip this function and go to Process_data")
                return
        h = temporary_sheet.shape[1]
        w = temporary_sheet.shape[2]
        
        try:
            #Try to check channels
            channels = temporary_sheet.shape[3]
        except:
            #If fails its grayscale or single channel array
            channels = 1
            
        del temporary_sheet

        if channels > 1 :
            x = np.zeros((0,h,w,channels) , dtype = np.uint8)
        elif channels == 1:
            x = np.zeros((0,h,w) , dtype = np.uint8)
        else:
            print("ERROR:   Channel number is incorrect, check file format (RGB/Grayscale)")
        
        print("Reducing amount of images in the class...\n")
        for class_name in Classes_list:
            print('Preparing '+str(class_name) + " class")
            Dictionary.append((Classes_list.index(class_name),class_name))
            directory = os.path.join(Data_directory , class_name)
            if reduced_class_shuffle:
                #Copy on write, file will be modified only if file needs to be modified without affecting original data. Memory efficient, but not as much as 'r'
                X_temp = np.load(directory , mmap_mode='c')
            else:
                #Read mode only, reduceds memory needed for operations
                X_temp = np.load(directory , mmap_mode='r')
            
            if samples_per_class is not None:
                X_temp = DataSets.Reduce_Img_Classification_Class_Size(X_temp , samples_per_class, reduced_class_shuffle)
            
            blank_class=np.zeros((len(X_temp),n_classes) , dtype = np.uint8)
            blank_class[:,Classes_list.index(class_name)] = 1
            #Adding class identificator to main set of classes [Y]
            ClassSet = np.concatenate((ClassSet, blank_class))
            x = np.concatenate((x, X_temp))
            del X_temp
 
        return x , ClassSet , Dictionary
            
                
    class Augmentation_Config:
        """
        Configuration class for data augmentation.
    
        Parameters:
        apply_flip (bool): If True, apply random left-right flips to images.
        apply_flip_up_down (bool): If True, apply random up-down flips to images.
        apply_brightness (bool): If True, apply random brightness adjustments.
        brightness_range (tuple): Range for brightness adjustment as (min, max).
        apply_contrast (bool): If True, apply random contrast adjustments.
        contrast_range (tuple): Range for contrast adjustment as (min, max).
        apply_saturation (bool): If True, apply random saturation adjustments.
        saturation_range (tuple): Range for saturation adjustment as (min, max).
        apply_hue (bool): If True, apply random hue adjustments.
        hue_delta (float): Max delta for hue adjustment.
        apply_crop (bool): If True, apply random cropping to images.
        crop_factor_range (tuple): Range for cropping factor as (min, max).
        apply_noise (bool): If True, add Gaussian noise to images.
        noise_std_range (tuple): Range for noise standard deviation as (min, max).
        """
    
        def __init__(self, 
                     apply_flip_left_right: bool = False, 
                     apply_flip_up_down: bool = False,  # New parameter for up-down flipping
                     apply_rotation: bool = False,
                     rotation_range: int = 15,
                     apply_brightness: bool = False, 
                     brightness_delta: float = 0.15, 
                     apply_contrast: bool = False, 
                     contrast_range: tuple = (0.9, 1.1), 
                     apply_saturation: bool = False,  # New parameter for saturation
                     saturation_range: tuple = (0.9, 1.1),  # Range for saturation adjustment
                     apply_hue: bool = False,  # New parameter for hue adjustment
                     hue_delta: float = 0.1,  # Max delta for hue adjustment
                     apply_crop: bool = False, 
                     crop_factor_range: tuple = (0.9, 1.0), 
                     apply_width_shift: bool = False,
                     width_shift_range: float = 0.1,
                     apply_height_shift: bool = False,
                     width_height_range: float = 0.1,
                     apply_noise: bool = False, 
                     noise_std_range: tuple = (0.0, 0.01)):
            
            self.apply_flip_left_right = apply_flip_left_right
            self.apply_flip_up_down = apply_flip_up_down
            self.apply_rotation = apply_rotation
            self.rotation_range = rotation_range
            self.apply_brightness = apply_brightness
            self.brightness_delta = brightness_delta
            self.apply_contrast = apply_contrast
            self.contrast_range = contrast_range
            self.apply_saturation = apply_saturation
            self.saturation_range = saturation_range
            self.apply_hue = apply_hue
            self.hue_delta = hue_delta
            self.apply_crop = apply_crop      
            self.crop_factor_range = crop_factor_range
            self.apply_width_shift = apply_width_shift
            self.width_shift_range = width_shift_range
            self.apply_height_shift = apply_height_shift
            self.height_shift_range = width_height_range
            self.apply_noise = apply_noise
            self.noise_std_range = noise_std_range  
    
        def __str__(self):
            t = Texttable()
            no_value = "----------"
            t.add_rows([
                ["Augmentation", 'Applied', 'Value'],
                ['Horizontal flip', self.apply_flip_left_right, no_value],
                ['Up-down flip', self.apply_flip_up_down, no_value],
                ['Rotation', self.apply_rotation, str(self.rotation_range)],
                ['Random brightness', self.apply_brightness, str(self.brightness_delta)],
                ['Random contrast', self.apply_contrast, str(self.contrast_range)],
                ['Random saturation', self.apply_saturation, str(self.saturation_range)],
                ['Random hue', self.apply_hue, self.hue_delta],
                ['Random crop', self.apply_crop, str(self.crop_factor_range)],
                ['Width_shift', self.apply_width_shift, str(self.width_shift_range)],
                ['Height_shift', self.apply_height_shift, str(self.height_shift_range)],
                ['Random noise', self.apply_noise, str(self.noise_std_range)],
            ])
            return t.draw()
    
        def get_dict(self):
            """
            Returns a dictionary of applied augmentations and their parameters.
            If an augmentation is not applied, it is excluded from the dictionary.
            """
            augmentations = {}
            
            if self.apply_flip_left_right:
                augmentations['Horizontal flip'] = "No params needed"
                
            if self.apply_flip_up_down:
                augmentations['Up-down flip'] = "No params needed"
                
            if self.apply_rotation:
                augmentations['Rotation'] = self.rotation_range
                
            if self.apply_brightness:
                augmentations['Random brightness'] = self.brightness_delta
                
            if self.apply_contrast:
                augmentations['Random contrast'] = self.contrast_range
                
            if self.apply_saturation:
                augmentations['Random saturation'] = self.saturation_range
                
            if self.apply_hue:
                augmentations['Random hue'] = self.hue_delta
                
            if self.apply_crop:
                augmentations['Random crop'] = self.crop_factor_range
                
            if self.apply_width_shift:
                augmentations['Width shift'] = self.width_shift_range

            if self.apply_height_shift:
                augmentations['Height shift'] = self.height_shift_range
                
            if self.apply_noise:
                augmentations['Random noise'] = self.noise_std_range
                
            return augmentations
    
    @tf.autograph.experimental.do_not_convert
    def Augment_dataset(image, label, config=None):
        # Set default config if none provided
        if config is None:
            config = DataSets.Augmentation_Config()
    
        # Random flip (left-right)
        if config.apply_flip_left_right:
            image = tf.image.random_flip_left_right(image)
    
        # Random flip (up-down)
        if config.apply_flip_up_down:
            image = tf.image.random_flip_up_down(image)
            
        # Random brightness
        if config.apply_brightness:
            delta = config.brightness_delta
            # Use delta directly for brightness adjustment
            image = tf.image.random_brightness(image, max_delta=delta)
    
        # Random contrast
        if config.apply_contrast:
            lower, upper = config.contrast_range
            image = tf.image.random_contrast(image, lower=lower, upper=upper)
    
        # Random saturation
        if config.apply_saturation:
            lower, upper = config.saturation_range
            image = tf.image.random_saturation(image, lower=lower, upper=upper)
    
        # Random hue
        if config.apply_hue:
            image = tf.image.random_hue(image, max_delta=config.hue_delta)
    
        # Random crop
        if config.apply_crop:
            # Generate a random crop factor within the specified range
            factor = tf.random.uniform((), 
                                       minval=config.crop_factor_range[0], 
                                       maxval=config.crop_factor_range[1])
        
            # Get dynamic shape of the image
            size = tf.shape(image)
            
            # Cast size to float32 before multiplication
            h = tf.cast(tf.cast(size[0], tf.float32) * factor, tf.int32)
            w = tf.cast(tf.cast(size[1], tf.float32) * factor, tf.int32)
        
            # Crop and resize
            image = tf.image.random_crop(image, size=(h, w, 3))  # Assuming RGB images
            image = tf.image.resize(image, (size[0], size[1]), method='bilinear')
            
        if config.apply_width_shift:
            shift_w = tf.cast(tf.cast(tf.shape(image)[1], tf.float32) * config.width_shift_range, tf.int32)
            direction_w = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32) * 2 - 1  # Random -1 or 1
            shift_w = shift_w * direction_w
            paddings = [[0, 0], [abs(shift_w), abs(shift_w)], [0, 0]]
            
            # Generate random boolean value
            #random_values = tf.random.uniform(shape=[], minval=0, maxval=1)
            #boolean_tensor = random_values > 0.5
            
            # Select padding mode based on the boolean value
            #if boolean_tensor:  # True: Use reflect padding
                #padded_image = tf.pad(image, paddings, mode='REFLECT')
            #else:  # False: Use constant padding with black background
            padded_image = tf.pad(image, paddings, mode='CONSTANT', constant_values=0)
            
            # Use tf.cond for offset_width
            offset_width = tf.cond(shift_w < 0, lambda: abs(shift_w), lambda: tf.constant(0, dtype=tf.int32))
            # Crop the image after padding
            image = tf.image.crop_to_bounding_box(
                padded_image, 
                offset_height=0, 
                offset_width = offset_width,
                target_height=tf.shape(image)[0], 
                target_width=tf.shape(image)[1]
            )
            
        if config.apply_height_shift:
            # Calculate shift for height
            shift_h = tf.cast(tf.cast(tf.shape(image)[0], tf.float32) * config.height_shift_range, tf.int32)
            direction_h = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32) * 2 - 1  # Random -1 or 1
            shift_h = shift_h * direction_h
            paddings = [[abs(shift_h), abs(shift_h)], [0, 0], [0, 0]]
            
            # Generate random boolean value
            #random_values = tf.random.uniform(shape=[], minval=0, maxval=1)
            #boolean_tensor = random_values > 0.5
            
            # Select padding mode based on the boolean value
            #if boolean_tensor:  # True: Use reflect padding
               # padded_image = tf.pad(image, paddings, mode='REFLECT')
            #else:  # False: Use constant padding with black background
            padded_image = tf.pad(image, paddings, mode='CONSTANT', constant_values=0)
            
            offset_height = tf.cond(shift_h < 0, lambda: abs(shift_h), lambda: tf.constant(0, dtype=tf.int32))

            # Crop the image after padding
            image = tf.image.crop_to_bounding_box(
                padded_image, 
                offset_height= offset_height,
                offset_width=0, 
                target_height=tf.shape(image)[0], 
                target_width=tf.shape(image)[1]
            )

    
        # Add noise
        if config.apply_noise:
            noise_std = tf.random.uniform(shape=[], 
                                           minval=config.noise_std_range[0], 
                                           maxval=config.noise_std_range[1])
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_std)
            image = tf.clip_by_value(image + noise, 0.0, 1.0)
    
    
        if config.apply_rotation:
            # Convert angle from degrees to radians
            angle = tf.random.uniform([], minval=-config.rotation_range, maxval=config.rotation_range)
            radians = tf.convert_to_tensor(angle * (np.pi / 180), dtype=tf.float32)
            
            # Get image dimensions
            height = tf.shape(image)[0]
            width = tf.shape(image)[1]
            channels =tf.shape(image)[2]
            
        
            # Compute the center of the image
            center_x = tf.cast(width // 2, tf.float32)
            center_y = tf.cast(height // 2, tf.float32)
        
            # Create a meshgrid of x and y coordinates
            x_indices, y_indices = tf.meshgrid(tf.range(width), tf.range(height), indexing='xy')
        
            # Flatten the grid
            x_indices = tf.reshape(x_indices, [-1])
            y_indices = tf.reshape(y_indices, [-1])
        
            # Compute the rotation matrix components
            cos_theta = tf.cos(radians)
            sin_theta = tf.sin(radians)
        
            # Apply the inverse rotation to map the output image back to the input image
            x_rotated = cos_theta * (tf.cast(x_indices, tf.float32) - center_x) + sin_theta * (tf.cast(y_indices, tf.float32) - center_y) + center_x
            y_rotated = -sin_theta * (tf.cast(x_indices, tf.float32) - center_x) + cos_theta * (tf.cast(y_indices, tf.float32) - center_y) + center_y
        
            # Round and clip the rotated coordinates to the valid range
            x_rotated = tf.cast(tf.round(x_rotated), tf.int32)
            y_rotated = tf.cast(tf.round(y_rotated), tf.int32)
        
            # Ensure the coordinates are within bounds
            x_rotated = tf.clip_by_value(x_rotated, 0, width - 1)
            y_rotated = tf.clip_by_value(y_rotated, 0, height - 1)
        
            # Gather pixel values from the original image
            indices = y_rotated * width + x_rotated
            
            reshaped_image = tf.reshape(image, [-1, channels])
        
            gathered_image = tf.gather(reshaped_image, indices)
            image = tf.reshape(gathered_image,(height,width,channels))
        
        return image, label





    
class ImageProcessing:
    
   
    def random_rotate_flip(img , mask = None):
        if random.choice([True, False]):
            #Horizontal rotate
            img = np.fliplr(img)
            if mask is not None:
                mask = np.fliplr(mask)
        if random.choice([True, False]):
            #Vertical rotate
            img = np.flipud(img)
            if mask is not None:
                mask = np.flipud(mask)
        random_angle = np.random.randint(0, 360)
        img = rotate(img, angle=random_angle, reshape=False, mode='constant', cval=0)
        if mask is not None:
            mask = rotate(mask, angle=random_angle, reshape=False, mode='constant', cval=0)
            
        if mask is not None:
            return img , mask
        else: 
            return img
    

    def add_gaussian_noise(img , strenght = 0.5):
        big_range = True
    
        if np.max(img) <=1:
            big_range = False
        
        
        std_dev = np.std(img)*0.3
        
        gauss = np.random.normal(0 , std_dev,  img.shape).astype(np.float16)
        
        gauss += img
        if big_range:
            gauss = np.clip(gauss ,0,255)
        else:
            gauss = np.clip(gauss,0,1)
        gauss = gauss.astype(img.dtype)
        
        return gauss
    
   
    def random_brightness(img, brightness_range=(0.7, 1.3)):
        big_range = True
        if np.max(img) <=1:
            big_range = False
        # Generate a random brightness factor within the specified range
        brightness_factor = np.random.uniform(*brightness_range)
        brightness_factor = float(brightness_factor)
        # Adjust brightness using cv2.multiply
        brightened_img = img*brightness_factor
        
        if big_range:
            # Clip values to ensure they are in the valid range [0, 1]
            brightened_img = np.clip(brightened_img, 0, 255).astype(img.dtype)
        else:
            brightened_img = np.clip(brightened_img, 0, 1).astype(img.dtype)
            
    
        return brightened_img
    
  
    def contour_mod(image , density = 2):
        gr = False
        if len(image.shape) == 2 or image.shape[2] == 1:
            gr = True
        big_range = True
        if np.max(image) <=1:
            big_range = False
        
        if not gr:
            img_mod = np.rollaxis(image, 2, 0)
            equ = []
            for channel in img_mod:
                if big_range:
                    equ.append(cv2.equalizeHist(np.array(channel*255 , dtype = np.uint8)))
                else:
                    equ.append(cv2.equalizeHist(np.array(channel , dtype = image.dtype)))
                
            equ = np.asarray(equ)  
            equ = np.rollaxis(equ , 0 , 3 )
            gray = cv2.cvtColor(equ, cv2.COLOR_RGB2GRAY) 
        else:
            if big_range:
                equ = np.array(cv2.equalizeHist(image) , dtype = np.uint8)
            else:
                equ = np.array(cv2.equalizeHist(image) , dtype = image.dtype)
                
            gray = equ
        
            
        
        blurr = cv2.blur(gray,(9,9))
        ret3,_ = cv2.threshold(blurr,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blurRGB = cv2.blur(np.array(image*255 , dtype = np.uint8) ,(5,5))
        edges = cv2.Canny(blurRGB , ret3*0.5 , ret3)
        edges = cv2.dilate(edges ,(3,3) , iterations = density)
        
        if big_range:
            edges = np.array(edges)
        else:
            edges = np.array(edges/255 , dtype = bool)
            
        feature_img = np.dstack((image, edges)).astype(image.dtype)
        return feature_img
     
        
    def denoise_img(img):
        img_type = img.dtype
        big_range = True
    
        if np.max(img) <=1:
            img = np.array(img*255 , dtype = np.uint8)
            big_range = False
            
        if len(img.shape) == 3 and img.shape[2] == 3 :
            img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 15)
            
        elif len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.fastNlMeansDenoising(img, None, 5, 7, 15)
            
        else:
            img = None
            print("Check if there is no multidimmensional array  or incorrect channel number for RGB/grayscale")
        
        
        if big_range:
            return np.array(img , dtype = img_type)
        else:
            return np.array(img/255 , dtype = img_type)
        










class General:
    def extract_type(data_type):
        # Extract the module and full class name
        module_name = data_type.__module__
        class_name = data_type.__name__
        
        # Construct and return the result
        return f"{module_name}.{class_name}"
 
    def save_function_as_string(func, save_path=None):
        # Create an empty dictionary if func is None
        if func is None:
            empty_data = {}
            # Save an empty JSON file if save_path is specified
            if save_path is not None:
                with open(save_path, "w") as json_file:
                    json.dump(empty_data, json_file)  # Write an empty dictionary to the file
            return empty_data
        
        # Get the source code of the function
        function_code = inspect.getsource(func)
        
        # Create a dictionary to hold the function code
        function_data = {
            "function_name": func.__name__,
            "function_code": function_code
        }
        
        # Convert the dictionary to a JSON string
        json_data = json.dumps(function_data, indent=4)
        
        # Save the JSON string to the specified file path if save_path is provided
        if save_path is not None:
            with open(save_path, "w") as json_file:
                json_file.write(json_data)
        
        # Return the function code as a string
        return function_code
 
    
    def save_custom_objects(model, custom_objects_path):
        custom_objects = {}
    
        # Function to determine if the object is user-defined
        def is_user_defined(obj):
            # Check if the object is callable and does not belong to the standard TensorFlow/Keras module
            return callable(obj) and not (obj.__module__.startswith('tensorflow.keras') or obj.__module__.startswith('keras'))
    
        # Detect custom layers and save only user-defined ones
        for layer in model.layers:
            if is_user_defined(layer.__class__):
                custom_objects[layer.__class__.__name__] = inspect.getsource(layer.__class__)
    
        # Detect custom loss, metrics, and optimizer
        if is_user_defined(model.loss):
            custom_objects[model.loss.__name__] = inspect.getsource(model.loss)
        if is_user_defined(model.optimizer):
            custom_objects[model.optimizer.__class__.__name__] = inspect.getsource(model.optimizer)
        for metric in model.metrics:
            if is_user_defined(metric):
                custom_objects[metric.__class__.__name__] = inspect.getsource(metric)
    
        # Serialize and save custom objects
        if custom_objects_path is not None:
            with open(custom_objects_path, "w") as f:
                json.dump(custom_objects, f, indent=4)  # Use indent for better readability
        else:
            return custom_objects


    
    def load_custom_objects(custom_objects_path):
        """Load custom objects from a JSON file."""
        with open(custom_objects_path, "r") as f:
            custom_objects_serialized = json.load(f)
    
        custom_objects = {}
        exec_globals = globals().copy()
        exec_globals.update({
            'tf': tf,
            'keras': tf.keras,
            'Layer': tf.keras.layers.Layer,
        })
    
        for name, source_code in custom_objects_serialized.items():
            # Dynamically execute the source code to recreate the custom function or class
            exec(source_code, exec_globals)
            custom_objects[name] = eval(name, exec_globals)
    
        return custom_objects
    
    def save_model_as_json(model, filename='model_architecture.json'):
        """
        Saves the model architecture as a JSON file.

        Args:
        model (tf.keras.Model): The Keras model to save.
        filename (str): The filename to save the JSON as.
        """
        model_json = model.to_json()
        with open(filename, "w") as json_file:
            json_file.write(model_json)


    def load_model_from_json(filename='model_architecture.json'):
        """
        Loads a Keras model from a JSON file.

        Args:
        filename (str): The JSON file containing the model architecture.

        Returns:
        tf.keras.Model: The loaded Keras model.
        """
        with open(filename, "r") as json_file:
            loaded_model_json = json_file.read()
        model = tf.keras.models.model_from_json(loaded_model_json)
        print(f"Model architecture loaded from {filename}")
        return model
    
    
    
    def compute_overtrain_metric(train_metric,val_metric, alpha = 2):
        otr = (train_metric - val_metric) / train_metric
        ofi = otr * math.exp(alpha * train_metric)
        normalized_ofi = ofi / math.exp(alpha)
        return normalized_ofi
    
    
    
    
    
    def compute_multiclass_roc_auc(y_true, y_pred):
        # Ensure y_true and y_pred have correct shapes
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f'Number of samples in y_true ({y_true.shape[0]}) does not match number of samples in y_pred ({y_pred.shape[0]})')

        n_classes = y_true.shape[1]

        # Compute the macro-average ROC AUC score
        roc_auc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')

        # Compute ROC AUC for each class
        roc_auc_per_class = []
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc_per_class.append(auc(fpr, tpr))

        return roc_auc, roc_auc_per_class
    
    
    def calculate_main_metrics(y_true_one_hot, y_pred_prob_one_hot):
        """
        Calculate various classification metrics from one-hot encoded true labels and predicted probabilities.
    
        Parameters:
        - y_true_one_hot: np.ndarray, shape (num_samples, num_classes)
          One-hot encoded true labels.
        - y_pred_prob: np.ndarray, shape (num_samples, num_classes)
          Predicted class probabilities from the model.
    
        Returns:
        - dict: Dictionary containing accuracy, precision, recall, f1_score, f2_score, f0.5_score, specificity, balanced_accuracy
        """
        # Convert one-hot encoded true labels to class labels
        y_true = np.argmax(y_true_one_hot, axis=1)
        # Convert predicted probabilities to class labels
        y_pred = np.argmax(y_pred_prob_one_hot, axis=1)
    
        # Calculate confusion matrix
        num_classes = y_pred_prob_one_hot.shape[1]
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    
        # Calculate specificity for each class
        specificity = []
        for i in range(num_classes):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity.append(tn / (tn + fp + 1e-10))  # Avoid division by zero
        specificity = np.mean(specificity)
    
    
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division = 0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division = 0)
        f2_score = General.calculate_fbeta_score(y_true, y_pred, beta=2)
        f0_5_score = General.calculate_fbeta_score(y_true, y_pred, beta=0.5)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'f2_score': f2_score,
            'f0_5_score': f0_5_score,
            'specificity': specificity,
            'balanced_accuracy': balanced_accuracy
        }
    
    # Calculate F-beta scores
    def calculate_fbeta_score(y_true, y_pred, beta):
        precision = precision_score(y_true, y_pred, average='macro', zero_division = 0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division = 0)
        fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-10)
        return fbeta
    
    
    
    def true_positives(y_true, y_pred):
        y_pred_rounded = np.round(np.clip(y_pred, 0, 1))
        tp = np.sum(y_true * y_pred_rounded)
        return int(tp) 
    
    def true_negatives(y_true, y_pred):
        y_pred_rounded = np.round(np.clip(y_pred, 0, 1))
        tn = np.sum((1 - y_true) * (1 - y_pred_rounded))
        return int(tn) 
    
    def false_positives(y_true, y_pred):
        y_pred_rounded = np.round(np.clip(y_pred, 0, 1))
        fp = np.sum((1 - y_true) * y_pred_rounded)
        return int(fp) 
    
    def false_negatives(y_true, y_pred):
        y_pred_rounded = np.round(np.clip(y_pred, 0, 1))
        fn = np.sum(y_true * (1 - y_pred_rounded))
        return int(fn) 
    
    
    def hash_string(input_string, hash_algorithm='sha256'):
        """
        Hashes a string using the specified hash algorithm.
        
        Args:
        - input_string (str): The string to hash.
        - hash_algorithm (str): The hashing algorithm to use (default: 'sha256').
                               Options include 'sha256', 'sha512', 'md5', etc.
        
        Returns:
        - str: Hexadecimal representation of the hashed value.
        """
        # Select the hash algorithm
        if hash_algorithm == 'sha256':
            hash_object = hashlib.sha256()
        elif hash_algorithm == 'sha512':
            hash_object = hashlib.sha512()
        elif hash_algorithm == 'md5':
            hash_object = hashlib.md5()
        else:
            raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")
    
        # Update hash object with the input string encoded as UTF-8
        hash_object.update(input_string.encode('utf-8'))
    
        # Get the hexadecimal representation of the hash digest
        hashed_string = hash_object.hexdigest()
        
        return hashed_string
    
    
    # Function to count trainable and non-trainable parameters
    def Count_parameters(model):
        trainable_count = int(
            np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
        )
        non_trainable_count = int(
            np.sum([tf.keras.backend.count_params(p) for p in model.non_trainable_weights])
        )
        return trainable_count, non_trainable_count
    

    
    def Load_model_check_training_progress(model , train , epochs_to_train, model_weights_directory, model_history_directory, custom_obj_directory):
        starting_epoch = None
        if train:
            try:
                Model_history = pd.read_csv(model_history_directory)
                starting_epoch = Model_history["epoch"].iloc[-1]+1
                try:
                    best_val_acc = Model_history["val_accuracy"].idxmax()
                    epoch_index = (Model_history['val_accuracy'] == Model_history["val_accuracy"][best_val_acc]).idxmax()
                    
                    best_val_loss = round(Model_history["val_loss"][best_val_acc],3)
                    best_val_acc = round(Model_history["val_accuracy"][best_val_acc],3)
                except:
                    print("Could not load model scores...")
        
                print("Found existing model trained for ",starting_epoch," epochs")
                try:
                    
                    print("Best model score aqcuired in ",Model_history["epoch"][epoch_index]," epoch\nVal_acc: ",best_val_acc,"\nVal_loss: ",best_val_loss,)
                except:
                    print("No score available")
                if starting_epoch == epochs_to_train:
                    print("\nTraining of this model is completed, do you want to load this model? \nType 'y' for yes and 'n' for no \n")
                else:
                    print("\nDo you want to continue training? \nType 'y' for yes and 'n' for no \n")
                user_input = input()
        
                while True:
                    if user_input.lower() =="y":
                        print("Continuing model training")
                        print("Loading trained weights to model...")
                        model.load_weights(model_weights_directory)
                        
                        #Switch to the model loading and optimizer there is better continuity
                        #Also maybe load custom functions so user does not have to define them every time
                        
                        
                        break
                    elif user_input.lower() =="n":
                        starting_epoch = 0
                        print("Model will be trained from scratch")
                        break
                    else:
                        print("Invalid input. Enter 'y' or 'n' ")
                
                

                
        
                
                
            except:
                starting_epoch = 0
                print("Could not load model weights")
                
           
        else:
        
            try:
                print("Loading trained weights to model and its training history...")
                Model_history = pd.read_csv(model_history_directory)
                #Trying to load model normally 
                try:
                    model = tf.keras.models.load_model(model_weights_directory)
                    #model.load_weights(model_weights_directory)
                    
               #Trying to load model with custom objects
                except:
                    custom = General.load_custom_objects(custom_obj_directory)
                    if len(custom) == 0:
                        custom = None
                    else:
                        print("Loaded custom objects into the model:\n")
                        print(custom)
                    model = tf.keras.models.load_model(model_weights_directory,custom_objects=custom)
                
                      
            except:
                starting_epoch = 0
                print("Could not load model weights and model history\n")
                print("Do you want to train model now?\nType 'y' for yes, 'n' for no\n")
                user_input = input()
                while True:
                    if user_input.lower() =="y":
                        train = True
                        print("Starting model training")
                        break
                    elif user_input.lower() =="n":
                        train = False
                        print("Model training skipped")
                        break
                    else:
                        print("Invalid input. Enter 'y' or 'n' ")
                    
                
        return model , train , starting_epoch
    
    
    def Model_training_history_plot_CSV(Model_training_history):
        try:
            Model_training_history.drop(columns=['epoch'], inplace=True)
        except:
            pass

        column_list = Model_training_history.columns.tolist()
        columns_score = [col for col in column_list if 'loss' not in col]
        val_score = [col for col in columns_score if 'val' in col][0]
        score = [col for col in columns_score if 'val' not in col][0]
        special_characters = ",[!@#$%^&*()+{}|:\"<>?-=[]\;',./"

        cleaned_score_title = ''.join(char for char in score if char not in special_characters)
        cleaned_val_score_title = ''.join(char for char in val_score if char not in special_characters)

        val_max = Model_training_history[val_score].idxmax()

        
        val_loss_min = Model_training_history["val_loss"].idxmin()






        plt.style.use('ggplot') 
        plt.figure(figsize = (10,5))
        plt.suptitle("Model training history")

        
        plt.subplot(1,2,1)
        plt.title(cleaned_score_title)
        plt.plot(Model_training_history[score] ,label = cleaned_score_title , c = "red" )
        plt.plot(Model_training_history[val_score] ,label = cleaned_val_score_title , c = "green" )
        plt.legend(loc = "lower right")
        plt.xlabel("Epoch")
        
        plt.axvline(x = val_max , color = "blue" , linestyle = "--" ,label = "test")
        plt.text(val_max, plt.ylim()[1], f'Max {cleaned_val_score_title} in: {val_max} epoch\n{cleaned_val_score_title}: {round(Model_training_history[cleaned_val_score_title][val_max],3)}', verticalalignment='bottom', horizontalalignment='left', color='blue')
        
        
        plt.subplot(1,2,2)
        plt.title("loss")
        plt.plot(Model_training_history["loss"] ,label = "loss" , c = "red" )
        plt.plot(Model_training_history["val_loss"] ,label = "val_loss" , c = "green" )
        plt.legend(loc = "upper right")
        plt.xlabel("Epoch")
        
        plt.axvline(x = val_loss_min , color = "blue" , linestyle = "--")
        plt.text(val_loss_min, plt.ylim()[1], f'Min val_loss in: {val_loss_min} epoch\nVal loss: {round(Model_training_history["val_loss"][val_loss_min],3)}', verticalalignment='bottom', horizontalalignment='left', color='blue')

        # Show the plot
        plt.show()
        plt.style.use('default') 


    def Conf_matrix_classification(y_test, y_pred, dictionary, title, normalize=False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        # Create confusion matrix and normalize it over predicted (columns)
        cm = skl.metrics.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        
        # Visualize confusion matrix
        classes = [x[1] for x in dictionary]
        classes = [x.split('.')[0] for x in classes]
        
        
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        cm = np.around(cm, 2)
        
        
        plt.imshow(cm, interpolation='nearest', cmap="BuGn")
        title = "Confusion matrix - "+title
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


    def OneHot_decode(labels,n_classes):
        y = np.eye(n_classes)[labels]
        return y
 
       
    def Stratification_test(labels):
        print("Dataset stratification test: \n")
        for i in range(labels.shape[1]):
            print("Class",i,"share: ")
            print("Training Set: ", round(sum(labels[:,i])/len(labels),2) )


    #Generating real samples from dataset
    def generate_real_samples(dataset, n_samples):
        """
        Taking n random samples from real dataset

        Args:
            dataset (array):   Dataset to take real samples from
            n_samples (int)    Amount of samples to take from dataset


        Returns:
            x (array): array of real samples from dataset
            y (array): array of ones (labels of real samples)
        """  
        #Generate random indexes
        idx = np.random.randint(0, len(dataset), n_samples)
        #Get random images from dataset
        x = dataset[idx]
        #generating labels for real class
        y = np.ones((n_samples, 1))
        return x, y   

        
    #Generating fake samples using noise and generator
    def generate_fake_samples(gan_generator, latent_dim, n_samples):
        """
        Generating n samples using generator
    
        Args:
            gan_generator (compiled model):     Trained generator model
            latent_dim (array):                 number of parameters to create random noise, it serves as input to the generator
            n_samples (int):                    Amount of samples to generate
    
    
        Returns:
            x (array): array of generated samples
            y (array): array of zeros (labels of fake samples)
        """ 
        #generate noise as input for generator
        noise = np.random.normal(0, 1, (n_samples, latent_dim))
        # predict outputs
        x = gan_generator.predict(noise)
        if len(x.shape) == 4 and x.shape[3]==1:
            x = np.squeeze(x, axis = -1)
        # create 'fake' class labels (0)
        y = np.zeros((n_samples, 1))
        return x, y
    

    def delete_files_in_folder(folder_path):
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # Check if the path is a file (not a subdirectory)
            if os.path.isfile(file_path):
                # Delete the file
                os.remove(file_path)
    
    
    # Function to create and save a GIF
    def create_gif(gif_array, gif_filepath, gif_height, gif_width, fps=5):
        if not isinstance(gif_array, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        if gif_array.dtype != np.uint8:
            raise ValueError("Array must be in np.uint8 format; [0-255] range")
        
        # Convert numpy arrays to PIL Images
        images = [Image.fromarray(frame) for frame in gif_array]
        images = [img.resize((gif_width,gif_height), Image.Resampling.NEAREST) for img in images]
        
        # Save the images as a GIF
        images[0].save(
            gif_filepath, 
            save_all=True, 
            append_images=images[1:], 
            duration=int(1000 / fps), 
            loop=0
        )    


    def create_image_grid(img_array, size, RGB = False):
        #To improve so it can handle rgb images as well, also describe required array configuration to work
        
        if len(img_array.shape) == 3 and not RGB:
            single_iteration = True
            
        elif len(img_array.shape) == 4 and RGB:
            single_iteration = True

        else:
            single_iteration = False
            
              
        try:
            if RGB:
                if single_iteration:
                    img_array = img_array[0:size**2,:,:,:]
                else:
                    img_array = img_array[:,0:size**2,:,:,:]
                
            else:
                if single_iteration:
                    img_array = img_array[0:size**2,:,:]
                else:
                    img_array = img_array[:,0:size**2,:,:]
                
        except:
            print("Size is too big for given array, increase number of samples or decrease size of grid")
            return
            
        # Calculate grid size (e.g., 5x5 for 25 images)
        if single_iteration:
            iterations = 1
            num_images = img_array.shape[0]
            cell_height = img_array.shape[1]
            cell_width = img_array.shape[2]

        else:
            iterations = img_array.shape[0]
            num_images = img_array.shape[1]
            cell_height = img_array.shape[2]
            cell_width = img_array.shape[3]
            
        grid_size = int(np.ceil(np.sqrt(num_images)))
        
        # Determine the size of each cell in the grid
        grid_array = []
        # Create an empty grid image
        for i in range(iterations):
            if RGB:
                grid_img = np.zeros((grid_size * cell_height, grid_size * cell_width,3), dtype=np.uint8)
            else:
                grid_img = np.zeros((grid_size * cell_height, grid_size * cell_width), dtype=np.uint8)
                
            
            for idx in range(num_images):
                row = idx // grid_size
                col = idx % grid_size
                if RGB:
                    if single_iteration:
                        grid_img[row * cell_height: (row + 1) * cell_height, col * cell_width: (col + 1) * cell_width,:] = img_array[idx]
                    else:
                        grid_img[row * cell_height: (row + 1) * cell_height, col * cell_width: (col + 1) * cell_width,:] = img_array[i][idx]
                else:
                    if single_iteration:
                        grid_img[row * cell_height: (row + 1) * cell_height, col * cell_width: (col + 1) * cell_width] = img_array[idx]
                    else:
                        grid_img[row * cell_height: (row + 1) * cell_height, col * cell_width: (col + 1) * cell_width] = img_array[i][idx]
            
            grid_array.append(grid_img)   
        grid_array = np.array(grid_array)
        
        if single_iteration:
            grid_array = grid_array[0]
            
        return grid_array    
    
    
    def Image_interpolation(generator, n_variations, steps_to_variation, is_grayscale = False, create_gif = False, gif_path = "Interpolated_Gif.gif", gif_scale = 1, gif_fps = 20):
        gen_img_list = []
        n_vectors = n_variations
        steps = steps_to_variation
        
        latent_dim = int(generator.input.shape[1])
        img_H = generator.output.shape[1]
        img_W = generator.output.shape[2]
        
        #Interpolated latent vectors for smooth transition effect
        latent_vectors = [np.random.randn(latent_dim) for _ in range(n_vectors-1)]
        latent_vectors.append(latent_vectors[0])
        interpolated_latent_vectors = []
        for i in range(len(latent_vectors)-1):
            for alpha in np.linspace(0, 1, steps, endpoint=False):
                interpolated_vector = latent_vectors[i] * (1 - alpha) + latent_vectors[i + 1] * alpha
                interpolated_latent_vectors.append(interpolated_vector)
        # Add the last vector to complete the sequence
        
        for vector in tqdm(interpolated_latent_vectors,desc = "Creating interpolation plot..."):
            r_vector = np.reshape(vector , (1,len(vector)))
            
            gen_img = generator.predict(r_vector , verbose = 0)


            if len(gen_img.shape) >= 4 and not is_grayscale:
                gen_img = np.reshape(gen_img,(img_H,img_W,3))
                
            if len(gen_img.shape) >= 3 and is_grayscale:
                gen_img = np.reshape(gen_img,(img_H,img_W))

            gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min())
            gen_img_list.append(gen_img)
            ##########
        
        
            
            
        gen_img_list = np.array(gen_img_list)
        #return gen_img_list
        if create_gif:
            try:
                General.create_gif(gif_array = (gen_img_list*255).astype(np.uint8),
                                      gif_filepath = gif_path,
                                      gif_height = int(gen_img_list.shape[1]*gif_scale) ,
                                      gif_width = int(gen_img_list.shape[2]*gif_scale),
                                      fps = gif_fps
                                      )
                print("Interpolation gif created!")
                
            except:
                print("Could not create gif file")
        #Plot
        
        def update_interpol(i):
            ax.clear()  # Clear the previous image
            if is_grayscale:
                ax.imshow(gen_img_list[i], cmap="gray")
            else:
                ax.imshow(gen_img_list[i])
                
            # Optionally, update the title
            ax.axis("off")
            plt.draw()
        
        # Create the figure and the axis
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        
        # Create the slider
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        inter_slider = Slider(ax_slider, 'Interpolation', 0, len(gen_img_list) - 1, valinit=0, valstep=1)
        
        # Update the plot when the slider value changes
        inter_slider.on_changed(update_interpol)
        
        # Initialize the first plot
        update_interpol(0)
        
        plt.show()
        
        return gen_img_list , inter_slider    
    
    
class Optimized:
        
    @nb.njit
    def calc_mean(x_train, DataType = np.float32):
        """ Calculate mean of the entire dataset """
        return np.mean(x_train)
    
    @nb.njit
    def calc_std(x_train, DataType = np.float32):
        """ Calculate standard deviation of the entire dataset """
        return np.std(x_train)
    
    @nb.njit
    def calc_channel_mean(x_train, DataType = np.float32):
        """ Calculate mean for each channel in the dataset """
        _, _, _, channels = x_train.shape
        Channel_mean = np.empty(channels, dtype=DataType)
        for c in range(channels):
            Channel_mean[c] = x_train[:, :, :, c].mean()
        return Channel_mean
    
    @nb.njit
    def calc_channel_std(x_train, DataType = np.float32):
        """ Calculate standard deviation for each channel in the dataset """
        _, _, _, channels = x_train.shape
        Channel_std = np.empty(channels, dtype=DataType)
        for c in range(channels):
            Channel_std[c] = x_train[:, :, :, c].std()
        return Channel_std
    
    @nb.njit
    def calc_data_maximum(x_train, DataType = np.float32):
        """ Calculate the maximum value in the dataset """
        return np.array([np.max(x_train)], dtype=DataType)
    
    @nb.njit
    def calc_data_minimum(x_train, DataType = np.float32):
        """ Calculate the minimum value in the dataset """
        return np.array([np.min(x_train)], dtype=DataType)
    
    @nb.njit
    def calc_channel_data_minimum(x_train, DataType = np.float32):
        """ Calculate the minimum value for each channel in the dataset """
        _, _, _, channels = x_train.shape
        Channel_data_minimum = np.empty(channels, dtype=DataType)
        for c in range(channels):
            Channel_data_minimum[c] = DataType(x_train[:, :, :, c].min())
        return Channel_data_minimum
    
    @nb.njit
    def calc_channel_data_maximum(x_train, DataType = np.float32):
        """ Calculate the maximum value for each channel in the dataset """
        _, _, _, channels = x_train.shape
        Channel_data_maximum = np.empty(channels, dtype=DataType)
        for c in range(channels):
            Channel_data_maximum[c] = DataType(x_train[:, :, :, c].max())
        return Channel_data_maximum
    
    @nb.njit
    def calc_median(x_train, DataType = np.float32):
        """ Calculate median of the entire dataset """
        return DataType(np.median(x_train))
    
    @nb.njit
    def calc_iqr(x_train, DataType = np.float32):
        """ Calculate the interquartile range (IQR) of the dataset """
        q1 = DataType(np.percentile(x_train, 25))
        q3 = DataType(np.percentile(x_train, 75))
        return q3 - q1
    
    @nb.njit
    def calc_channel_median(x_train, DataType = np.float32):
        """ Calculate median for each channel in the dataset """
        _, _, _, channels = x_train.shape
        Channel_median = np.empty(channels, dtype=DataType)
        for c in range(channels):
            Channel_median[c] = np.median(x_train[:, :, :, c])
        return Channel_median
    
    @nb.njit
    def calc_channel_iqr(x_train, DataType = np.float32):
        """ Calculate the interquartile range (IQR) for each channel in the dataset """
        _, _, _, channels = x_train.shape
        Channel_IQR = np.empty(channels, dtype = DataType)
        for c in range(channels):
            q1 = np.percentile(x_train[:, :, :, c], 25)
            q3 = np.percentile(x_train[:, :, :, c], 75)
            Channel_IQR[c] = q3 - q1
        return Channel_IQR    
    
    
    
    
    
    
    
    
    
    