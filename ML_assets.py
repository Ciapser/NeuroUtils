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
import math
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Activation, Add, Input, GlobalAveragePooling2D, Reshape, Multiply, Dense, SpatialDropout2D
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DataSets:
    #Scenario 1
    #If it is classification data
    ################################################
    
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
        
       
        
            
    def Load_And_Merge_DataSet(Data_directory , samples_per_class = None):
        #############
        #Second part



        
        Classes_list  = os.listdir(Data_directory)
        n_classes = len(Classes_list)
        ClassSet=np.zeros((0,n_classes) , dtype = np.uint8)
        Dictionary = []
        
        temporary_sheet = np.load(os.path.join(Data_directory , Classes_list[0]) , mmap_mode='r')
        h = temporary_sheet.shape[1]
        w = temporary_sheet.shape[2]
        channels = temporary_sheet.shape[3]
        del temporary_sheet

        if channels == 3 :
            x = np.zeros((0,h,w,3) , dtype = np.uint8)
        elif channels == 1:
            x = np.zeros((0,h,w,1) , dtype = np.uint8)
        else:
            print("ERROR:   Channel number is incorrect, check file format (RGB/Grayscale)")
        
        print("Reducing amount of images in the class...\n")
        for class_name in Classes_list:
            print('Preparing '+str(class_name) + " class")
            Dictionary.append((Classes_list.index(class_name),class_name))
            directory = os.path.join(Data_directory , class_name)
            X_temp = np.load(directory , mmap_mode='r')

            if samples_per_class is not None:
                X_temp = DataSets.Reduce_Img_Classification_Class_Size(X_temp , samples_per_class)
            
            blank_class=np.zeros((len(X_temp),n_classes) , dtype = np.uint8)
            blank_class[:,Classes_list.index(class_name)] = 1
            #Adding class identificator to main set of classes [Y]
            ClassSet = np.concatenate((ClassSet, blank_class))
            x = np.concatenate((x, X_temp))
            del X_temp
 
        return x , ClassSet , Dictionary
            
            
              


    
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
    
    def Load_model_check_training_progress(model , train , model_weights_directory, model_history_directory):
        starting_epoch = None
        if train:
            try:
                Model_history = pd.read_csv(model_history_directory)
                starting_epoch = Model_history["epoch"].iloc[-1]
                best_val_acc = Model_history["val_accuracy"].idxmax()
                
                best_val_loss = round(Model_history["val_loss"][best_val_acc],3)
                best_val_acc = round(Model_history["val_accuracy"][best_val_acc],3)
        
                print("Found existing model trained for ",starting_epoch," epochs")
                print("Best model score aqcuired in ",starting_epoch," epoch\nVal_acc: ",best_val_acc,"\nVal_loss: ",best_val_acc,)
                
                print("Do you want to continue training? \nType 'y' for yes and 'n' for no \n")
                user_input = input()
        
                while True:
                    if user_input.lower() =="y":
                        print("Continuing model training")
                        print("Loading trained weights to model...")
                        model.load_weights(model_weights_directory)
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
                model.load_weights(model_weights_directory)
                
                
                      
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


    def Conf_matrix_classification(y_test, y_pred, dictionary, normalize=False):
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
        plt.title("Confusion matrix")
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


    def OneHot_decode(labels):
        w = list(set(labels))
        w = len(w)
        y = np.eye(w)[labels]
        
        return y
        

        
class Architectures():
    
    class Img_Classification():
        
        @staticmethod
        def AnimalNet_v64(shape , n_classes):
            img_H , img_W , channels = shape
            #Functions of network:
                
            def Swish(x):
                return x * tf.nn.sigmoid(x)    
                
                
            def cnn_block(input_layer , expand_filters , squeeze_filters = None ,kernel_size = 3, block_layers=3):
                if squeeze_filters is None:
                    squeeze_filters = expand_filters //4
                   
                x = input_layer
                for i in range(block_layers):
                    x_origin = x

                    x = tf.keras.layers.Conv2D(expand_filters, (1, 1), kernel_initializer='glorot_uniform', padding='same')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = Swish(x)
                        
                    
                    x = tf.keras.layers.DepthwiseConv2D((kernel_size,kernel_size), padding = 'same')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = Swish(x)
                    
                    x = tf.keras.layers.Conv2D(squeeze_filters , (1,1), padding = 'same')(x)
                    x = tf.keras.layers.BatchNormalization()(x)


                
                    x = tf.keras.layers.concatenate([x , x_origin])
                    #x = tf.keras.layers.Add()([x , x_origin])
                
                if i >= block_layers-1:
                    # Ensure spatial dimensions match before concatenation
                    #input_layer = tf.keras.layers.Conv2D(filters //2, (1, 1), padding='same')(input_layer)            
                    x_merged = tf.keras.layers.concatenate([x , input_layer]) 
                    
                    x_merged = tf.keras.layers.Conv2D(expand_filters, (1, 1), padding='same')(x_merged)
                    x_merged = tf.keras.layers.BatchNormalization()(x_merged)
                    x_merged = Swish(x_merged)
                    
                    x_merged = tf.keras.layers.DepthwiseConv2D((kernel_size,kernel_size), padding = 'same')(x_merged)
                    x_merged = tf.keras.layers.BatchNormalization()(x_merged)
                    x_merged = Swish(x_merged)

                    x_merged = tf.keras.layers.Conv2D(squeeze_filters, (1, 1), padding='same')(x_merged)
                    x_merged = tf.keras.layers.BatchNormalization()(x_merged)
                    x_merged = Swish(x_merged)
                    
                    x_merged = tf.keras.layers.SpatialDropout2D(0.2)(x_merged)
                
                return x_merged
            


            #Inputs
            inputs = tf.keras.layers.Input((img_H, img_W, channels))
            #########################################################
            #########################################################
            
            p0 = tf.keras.layers.Conv2D(48,(7,7) , padding = 'same')(inputs)
            p0 = tf.keras.layers.BatchNormalization()(p0)
            p0 = Swish(p0)
            
            
            
            
            
            
            d1 = cnn_block(p0 , 48  , kernel_size = 3 , block_layers = 3)
            #d1 = tf.keras.layers.MaxPooling2D((2,2))(d1)

            
            d2 = cnn_block(d1,64  , kernel_size = 3 , block_layers = 5)
            #d2 = tf.keras.layers.MaxPooling2D((2,2))(d2)

            
            d3 = cnn_block(d2,96  , kernel_size = 3 , block_layers = 5)
            d3 = tf.keras.layers.MaxPooling2D((2,2))(d3)

            d4 = cnn_block(d3,192  , kernel_size = 3 , block_layers = 3)
            #d4 = tf.keras.layers.MaxPooling2D((2,2))(d4)

            d5 = cnn_block(d4 , 256  , kernel_size = 3 , block_layers = 3)
            
            d6 = cnn_block(d5 , 384  , kernel_size = 3 , block_layers = 4)




            #d4 = tf.keras.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(d4)
            
            e2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(p0)
           # e2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(4, 4), padding='same')(e2)
            e2 = tf.keras.layers.BatchNormalization()(e2)
            e2 = Swish(e2)
            
            
            e1 = tf.keras.layers.concatenate([d6,e2])
            
            e1 = cnn_block(e1 , 128 , 64 , kernel_size = 3 , block_layers = 2)

            
            
            
            e0 = tf.keras.layers.GlobalAveragePooling2D()(e1)
            
            
            #########################################################
            #########################################################
            #Outputs 
            outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(e0)
            # Define the model
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            #Return model
            return model

        @staticmethod
        def AnimalNet_v32(shape , n_classes):
            img_H , img_W , channels = shape
            #Functions of network:
                
            def Swish(x):
                return x * tf.nn.sigmoid(x)    
                
                
            def cnn_block(input_layer , expand_filters , squeeze_filters = None ,kernel_size = 3, block_layers=3):
                if squeeze_filters is None:
                    squeeze_filters = expand_filters //4
                   
                x = input_layer
                for i in range(block_layers):
                    x_origin = x

                    x = tf.keras.layers.Conv2D(expand_filters, (1, 1), padding='same')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = Swish(x)
                        
                    
                    x = tf.keras.layers.DepthwiseConv2D((kernel_size,kernel_size), padding = 'same')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = Swish(x)
                    
                    x = tf.keras.layers.Conv2D(squeeze_filters , (1,1), padding = 'same')(x)
                    x = tf.keras.layers.BatchNormalization()(x)


                    
                    x = tf.keras.layers.concatenate([x , x_origin])
                    x = tf.keras.layers.SpatialDropout2D(0.1)(x)
                    #x = tf.keras.layers.Add()([x , x_origin])
                
                if i >= block_layers-1:
                    # Ensure spatial dimensions match before concatenation
                    #input_layer = tf.keras.layers.Conv2D(filters //2, (1, 1), padding='same')(input_layer)            
                    x_merged = tf.keras.layers.concatenate([x , input_layer]) 
                    
                    x_merged = tf.keras.layers.Conv2D(expand_filters, (1, 1), padding='same')(x_merged)
                    x_merged = tf.keras.layers.BatchNormalization()(x_merged)
                    x_merged = Swish(x_merged)
                    
                    x_merged = tf.keras.layers.DepthwiseConv2D((kernel_size,kernel_size), padding = 'same')(x_merged)
                    x_merged = tf.keras.layers.BatchNormalization()(x_merged)
                    x_merged = Swish(x_merged)

                    x_merged = tf.keras.layers.Conv2D(squeeze_filters, (1, 1), padding='same')(x_merged)
                    x_merged = tf.keras.layers.BatchNormalization()(x_merged)
                    x_merged = Swish(x_merged)
                    
                    x_merged = tf.keras.layers.SpatialDropout2D(0.1)(x_merged)
                
                return x
            


            #Inputs
            inputs = tf.keras.layers.Input((img_H, img_W, channels))
            #########################################################
            #########################################################
            
            p0 = tf.keras.layers.Conv2D(48,(5,5) , padding = 'same')(inputs)
            p0 = tf.keras.layers.BatchNormalization()(p0)
            p0 = Swish(p0)
            
            
            
            
            
            
            d1 = cnn_block(p0 , 48  , kernel_size = 3 , block_layers = 3)
            
            d2 = cnn_block(d1,64  , kernel_size = 3 , block_layers = 5)
            
            d3 = cnn_block(d2,96  , kernel_size = 3 , block_layers = 5)
            d3 = tf.keras.layers.MaxPooling2D((2,2))(d3)
            
            d4 = cnn_block(d3,192  , kernel_size = 3 , block_layers = 3)
            
            d5 = cnn_block(d4 , 256  , kernel_size = 3 , block_layers = 3)
            d5 = tf.keras.layers.MaxPooling2D((2,2))(d5)
            
            d6 = cnn_block(d5 , 384  , kernel_size = 3 , block_layers = 4)


            
            e0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(4, 4), padding='same')(p0)
          
            e0 = tf.keras.layers.BatchNormalization()(e0)
            e0 = Swish(e0)
            
            
            e1 = tf.keras.layers.concatenate([d6,e0])
            
            e1 = cnn_block(e1 , 128 , 64 , kernel_size = 3 , block_layers = 2)

            
            
            
            e2 = tf.keras.layers.GlobalAveragePooling2D()(e1)
            
            e3 = tf.keras.layers.Dense(256)(e2)
            e3 = Swish(e3)
            e3 = tf.keras.layers.BatchNormalization()(e3)
            e3 = tf.keras.layers.Dropout(0.2)(e3)
            
            e4 = tf.keras.layers.Dense(128)(e3)
            e4 = Swish(e4)
            e4 = tf.keras.layers.BatchNormalization()(e4)
            e4 = tf.keras.layers.Dropout(0.5)(e4)
            
            e5 = tf.keras.layers.Dense(64)(e4)
            e5 = Swish(e5)
            e5 = tf.keras.layers.BatchNormalization()(e5)
            e5 = tf.keras.layers.Dropout(0.4)(e5)
            
            
            #########################################################
            #########################################################
            #Outputs 
            outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(e5)
            # Define the model
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            #Return model
            return model
        
        @staticmethod
        def AnimalNet_v32_01(shape , n_classes):
            img_H , img_W , channels = shape
            #Functions of network:
                
            def Swish(x):
                return x * tf.nn.sigmoid(x)    
                
                
            def cnn_block(input_layer , expand_filters , squeeze_filters = None ,kernel_size = 3, block_layers=3):
                if squeeze_filters is None:
                    squeeze_filters = expand_filters //4
                   
                x = input_layer
                for i in range(block_layers):
                    x_origin = x

                    x = tf.keras.layers.Conv2D(expand_filters, (1, 1), padding='same')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = Swish(x)
                        
                    
                    x = tf.keras.layers.DepthwiseConv2D((kernel_size,kernel_size), padding = 'same')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = Swish(x)
                    
                    x = tf.keras.layers.Conv2D(squeeze_filters , (1,1), padding = 'same')(x)
                    x = tf.keras.layers.BatchNormalization()(x)


                    
                    x = tf.keras.layers.concatenate([x , x_origin])
                    x = tf.keras.layers.SpatialDropout2D(0.1)(x)
                    #x = tf.keras.layers.Add()([x , x_origin])
                
                if i >= block_layers-1:
                    # Ensure spatial dimensions match before concatenation
                    #input_layer = tf.keras.layers.Conv2D(filters //2, (1, 1), padding='same')(input_layer)            
                    x_merged = tf.keras.layers.concatenate([x , input_layer]) 
                    
                    x_merged = tf.keras.layers.Conv2D(expand_filters, (1, 1), padding='same')(x_merged)
                    x_merged = tf.keras.layers.BatchNormalization()(x_merged)
                    x_merged = Swish(x_merged)
                    
                    x_merged = tf.keras.layers.DepthwiseConv2D((kernel_size,kernel_size), padding = 'same')(x_merged)
                    x_merged = tf.keras.layers.BatchNormalization()(x_merged)
                    x_merged = Swish(x_merged)

                    x_merged = tf.keras.layers.Conv2D(squeeze_filters, (1, 1), padding='same')(x_merged)
                    x_merged = tf.keras.layers.BatchNormalization()(x_merged)
                    x_merged = Swish(x_merged)
                    
                    x_merged = tf.keras.layers.SpatialDropout2D(0.1)(x_merged)
                
                return x_merged
            


            #Inputs
            inputs = tf.keras.layers.Input((img_H, img_W, channels))
            #########################################################
            #########################################################
            
            p0 = tf.keras.layers.Conv2D(64,(5,5) , padding = 'same')(inputs)
            p0 = tf.keras.layers.BatchNormalization()(p0)
            p0 = Swish(p0)
            
            
            
            
            
            
            d1 = cnn_block(p0 , 64  , kernel_size = 3 , block_layers = 3)
            
            d2 = cnn_block(d1,96  , kernel_size = 3 , block_layers = 5)
            d2 = tf.keras.layers.MaxPooling2D((2,2))(d2)
            
            d3 = cnn_block(d2,128  , kernel_size = 3 , block_layers = 5)
            #d3 = tf.keras.layers.MaxPooling2D((2,2))(d3)
            
            d4 = cnn_block(d3,192  , kernel_size = 3 , block_layers = 3)
            
            d5 = cnn_block(d4 , 256  , kernel_size = 3 , block_layers = 3)
            d5 = tf.keras.layers.MaxPooling2D((2,2))(d5)
            
            d6 = cnn_block(d5 , 512  , kernel_size = 3 , block_layers = 4)


            
            e0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(4, 4), padding='same')(p0)
          
            e0 = tf.keras.layers.BatchNormalization()(e0)
            e0 = Swish(e0)
            
            
            e1 = tf.keras.layers.concatenate([d6,e0])
            
            e1 = cnn_block(e1 , 256 , 64 , kernel_size = 3 , block_layers = 2)

            
            
            
            e2 = tf.keras.layers.GlobalAveragePooling2D()(e1)
            
            e3 = tf.keras.layers.Dense(1024)(e2)
            e3 = Swish(e3)
            e3 = tf.keras.layers.BatchNormalization()(e3)
            e3 = tf.keras.layers.Dropout(0.2)(e3)
            
            e4 = tf.keras.layers.Dense(512)(e3)
            e4 = Swish(e4)
            e4 = tf.keras.layers.BatchNormalization()(e4)
            e4 = tf.keras.layers.Dropout(0.5)(e4)
            
            e5 = tf.keras.layers.Dense(256)(e4)
            e5 = Swish(e5)
            e5 = tf.keras.layers.BatchNormalization()(e5)
            e5 = tf.keras.layers.Dropout(0.4)(e5)
            
            
            #########################################################
            #########################################################
            #Outputs 
            outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(e5)
            # Define the model
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            #Return model
            return model


        @staticmethod
        def MobileNet_v2(shape , n_classes, alpha = 1):
            img_H , img_W , channels = shape
            #Functions of network:
                
            def Swish(x):
                return x * tf.nn.sigmoid(x)   
            
            def relu6(x):
                return min(max(0, x), 6)
            
            
            def inv_residual_block(x , filters , t = 6 , s = 1 ):

                squeeze = filters//t
                

                # Expansion phase: 1x1 convolution to increase channel dimensionality
                m = tf.keras.layers.Conv2D(filters, (1, 1))(x)
                m = tf.keras.layers.BatchNormalization()(m)
                m = Swish(m)
                
                # Depthwise convolution phase
                m = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(s, s), padding='same')(m)
                m = tf.keras.layers.BatchNormalization()(m)
                m = Swish(m)
                
                # Squeeze phase: 1x1 convolution to decrease channel dimensionality
                m = tf.keras.layers.Conv2D(squeeze, (1, 1))(m)
                m = tf.keras.layers.BatchNormalization()(m)
                
                b = tf.keras.layers.Conv2D(squeeze, (1, 1))(x)
                if s == 1:
                    final = tf.keras.layers.Add()([m,b])
                    return final
                else:
                    return m
            
            def bottleneck(x , t , c , n , s):
                for i in range(n):
                    if s >1:
                        x = inv_residual_block(x , c , t , s)
                        s = 1
                    else:
                        x = inv_residual_block(x , c , t , s)
                        
                return x
                        


            inputs = tf.keras.layers.Input((img_H, img_W, channels))
            
            c0 = tf.keras.layers.Conv2D(32*alpha, (3,3),strides=(2,2), padding="same")(inputs)
            
            b1 = bottleneck(c0 , t=1 , c=int(16*alpha) , n=1 , s=1)
            b1 = tf.keras.layers.SpatialDropout2D(0.1)(b1)
            
            b2 = bottleneck(b1 , t=6 , c=int(24*alpha) , n=2 , s=2)
            b2 = tf.keras.layers.SpatialDropout2D(0.1)(b2)
            
            b3 = bottleneck(b2 , t=6 , c=int(32*alpha) , n=3 , s=2)
            b3 = tf.keras.layers.SpatialDropout2D(0.1)(b3)
            
            b4 = bottleneck(b3 , t=6 , c=int(64*alpha) , n=4 , s=2)
            b4 = tf.keras.layers.SpatialDropout2D(0.1)(b4)
            
            b5 = bottleneck(b4 , t=6 , c=int(96*alpha) , n=3 , s=1)
            b5 = tf.keras.layers.SpatialDropout2D(0.1)(b5)
            
            b6 = bottleneck(b5 , t=6 , c=int(160*alpha) , n=3 , s=2)
            b6 = tf.keras.layers.SpatialDropout2D(0.1)(b6)
            
            b7 = bottleneck(b6 , t=6 , c=int(320*alpha) , n=1 , s=1)
            b7 = tf.keras.layers.SpatialDropout2D(0.1)(b7)
            
            c8 = tf.keras.layers.Conv2D(int(1280*alpha), (1,1), padding="same")(b7)
            c8 = Swish(c8)
            
            a9 = tf.keras.layers.GlobalAveragePooling2D()(c8)
            
            
            d0 = tf.keras.layers.Dense(int(256*alpha))(a9)
            d0 = Swish(d0)
            d0 = tf.keras.layers.BatchNormalization()(d0)
            d0 = tf.keras.layers.Dropout(0.1)(d0)
            
            d0 = tf.keras.layers.Dense(int(128*alpha))(d0)
            d0 = Swish(d0)
            d0 = tf.keras.layers.BatchNormalization()(d0)
            d0 = tf.keras.layers.Dropout(0.15)(d0)
            
            d0 = tf.keras.layers.Dense(int(64*alpha))(d0)
            d0 = Swish(d0)
            d0 = tf.keras.layers.BatchNormalization()(d0)
            d0 = tf.keras.layers.Dropout(0.1)(d0)
            

            
            
            outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(d0)

            model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
            return model


        @staticmethod
        def MobileNet_v2_corrected(shape , n_classes, alpha = 1):
            img_H , img_W , channels = shape
            #Functions of network:
                
            def Swish(x):
                return x * tf.nn.sigmoid(x)   
            
            def relu6(x):
                return min(max(0, x), 6)
            
            
            def inv_residual_block(x , filters , t = 6 , s = 1 ):

                squeeze = filters
                expand = filters*t
                

                # Expansion phase: 1x1 convolution to increase channel dimensionality
                m = tf.keras.layers.Conv2D(expand, (1, 1))(x)
                m = tf.keras.layers.BatchNormalization()(m)
                m = Swish(m)
                
                # Depthwise convolution phase
                m = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(s, s), padding='same')(m)
                m = tf.keras.layers.BatchNormalization()(m)
                m = Swish(m)
                
                # Squeeze phase: 1x1 convolution to decrease channel dimensionality
                m = tf.keras.layers.Conv2D(squeeze, (1, 1))(m)
                m = tf.keras.layers.BatchNormalization()(m)
                
                b = tf.keras.layers.Conv2D(squeeze, (1, 1))(x)
                if s == 1:
                    final = tf.keras.layers.Add()([m,b])
                    return final
                else:
                    return m
            
            def bottleneck(x , t , c , n , s):
                for i in range(n):
                    if s >1:
                        x = inv_residual_block(x , c , t , s)
                        s = 1
                    else:
                        x = inv_residual_block(x , c , t , s)
                        
                return x
                        


            inputs = tf.keras.layers.Input((img_H, img_W, channels))
            
            c0 = tf.keras.layers.Conv2D(32*alpha, (3,3),strides=(2,2), padding="same")(inputs)
            
            b1 = bottleneck(c0 , t=1 , c=int(16*alpha) , n=1 , s=1)
            b1 = tf.keras.layers.SpatialDropout2D(0.1)(b1)
            
            b2 = bottleneck(b1 , t=6 , c=int(24*alpha) , n=2 , s=2)
            b2 = tf.keras.layers.SpatialDropout2D(0.1)(b2)
            
            b3 = bottleneck(b2 , t=6 , c=int(32*alpha) , n=3 , s=2)
            b3 = tf.keras.layers.SpatialDropout2D(0.1)(b3)
            
            b4 = bottleneck(b3 , t=6 , c=int(64*alpha) , n=4 , s=2)
            b4 = tf.keras.layers.SpatialDropout2D(0.1)(b4)
            
            b5 = bottleneck(b4 , t=6 , c=int(96*alpha) , n=3 , s=1)
            b5 = tf.keras.layers.SpatialDropout2D(0.1)(b5)
            
            b6 = bottleneck(b5 , t=6 , c=int(160*alpha) , n=3 , s=2)
            b6 = tf.keras.layers.SpatialDropout2D(0.1)(b6)
            
            b7 = bottleneck(b6 , t=6 , c=int(320*alpha) , n=1 , s=1)
            b7 = tf.keras.layers.SpatialDropout2D(0.1)(b7)
            
            c8 = tf.keras.layers.Conv2D(int(1280*alpha), (1,1), padding="same")(b7)
            c8 = Swish(c8)
            
            a9 = tf.keras.layers.GlobalAveragePooling2D()(c8)
            
            
            d0 = tf.keras.layers.Dense(int(256*alpha))(a9)
            d0 = Swish(d0)
            d0 = tf.keras.layers.BatchNormalization()(d0)
            d0 = tf.keras.layers.Dropout(0.1)(d0)
            
            d0 = tf.keras.layers.Dense(int(128*alpha))(d0)
            d0 = Swish(d0)
            d0 = tf.keras.layers.BatchNormalization()(d0)
            d0 = tf.keras.layers.Dropout(0.15)(d0)
            
            d0 = tf.keras.layers.Dense(int(64*alpha))(d0)
            d0 = Swish(d0)
            d0 = tf.keras.layers.BatchNormalization()(d0)
            d0 = tf.keras.layers.Dropout(0.1)(d0)
            

            
            
            outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(d0)

            model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
            return model    

        @staticmethod
        def StupidNet(shape , n_classes):
            img_H , img_W , channels = shape


            
            inputs = tf.keras.layers.Input((img_H, img_W, channels))

            c1 = tf.keras.layers.Conv2D(64, (3,3), padding="same")(inputs)
            c1 = tf.keras.layers.BatchNormalization()(c1)
            c1 = tf.keras.layers.LeakyReLU()(c1)
            c1 = tf.keras.layers.Dropout(0.05)(c1)
            c1 = tf.keras.layers.MaxPooling2D((2,2))(c1)


            c2 = tf.keras.layers.Conv2D(128, (3,3), padding="same")(c1)
            c2 = tf.keras.layers.BatchNormalization()(c2)
            c2 = tf.keras.layers.LeakyReLU()(c2)
            c2 = tf.keras.layers.Dropout(0.05)(c2)
            #c2 = tf.keras.layers.MaxPooling2D((3,3))(c2)


            c3 = tf.keras.layers.Conv2D(256, (3,3), padding ="same")(c2)
            c3 = tf.keras.layers.BatchNormalization()(c3)
            c3 = tf.keras.layers.LeakyReLU()(c3)
            c3 = tf.keras.layers.Dropout(0.05)(c3)
           # c3 = tf.keras.layers.MaxPooling2D((2,2))(c3)


            k1 = tf.keras.layers.Conv2D(128, (3,3), padding ="same")(c3)
            k1 = tf.keras.layers.BatchNormalization()(k1)
            k1 = tf.keras.layers.LeakyReLU()(k1)
            k1 = tf.keras.layers.Dropout(0.05)(k1)
            #k1 = tf.keras.layers.MaxPooling2D((2,2))(k1)


            k2 = tf.keras.layers.Conv2D(32, (3,3), padding ="same")(k1)
            k2 = tf.keras.layers.BatchNormalization()(k2)
            k2 = tf.keras.layers.LeakyReLU()(k2)
            k2 = tf.keras.layers.Dropout(0.05)(k2)
            k2 = tf.keras.layers.MaxPooling2D((2,2))(k2)


            #c4 = tf.keras.layers.Flatten()(k2)
            c4 = tf.keras.layers.GlobalAveragePooling2D()(k2)
            
            c4 = tf.keras.layers.Dense(256,activation="relu")(c4)
            c4 = tf.keras.layers.Dropout(0.2)(c4)


            c5 = tf.keras.layers.Dense(128,activation="relu")(c4)
            c5 = tf.keras.layers.Dropout(0.5)(c5)


            c6 = tf.keras.layers.Dense(128,activation="relu")(c5)
            c6 = tf.keras.layers.Dropout(0.5)(c6)


            c7 = tf.keras.layers.Dense(64,activation="relu")(c6)
            c7 = tf.keras.layers.Dropout(0.4)(c7)





            ##############
                 
            #Okreslenie wartosci wyjsciowych

            outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(c6)

            model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
            return model



        @staticmethod
        def ResNet_own(shape , n_classes):
            img_H , img_W , channels = shape
            
            def Swish(x):
                return x * tf.nn.sigmoid(x)  

            
            def res_small_block(x , f , s , drop):
                if s>1:
                    r = tf.keras.layers.Conv2D(f , (3,3) , strides = (s,s) , padding = "same")(x)
                    r = tf.keras.layers.BatchNormalization()(r)
                    r = Swish(r)
                    
                    r = tf.keras.layers.Conv2D(f , (3,3) , strides = (1,1) , padding = "same")(r)
                    r = tf.keras.layers.BatchNormalization()(r)
                    r = Swish(r)
                    
                else:
                    r = tf.keras.layers.Conv2D(f , (3,3) , strides = (1,1) , padding = "same")(x)
                    r = tf.keras.layers.BatchNormalization()(r)
                    r = Swish(r)
                    
                    r = tf.keras.layers.Conv2D(f , (3,3) , strides = (1,1) , padding = "same")(r)
                    r = tf.keras.layers.BatchNormalization()(r)
                    r = Swish(r)
                    
                

                
                if s == 1:
                    r = tf.keras.layers.Add()([x,r])
                    r = tf.keras.layers.SpatialDropout2D(drop)(r)
                    return r
                else:
                    r = tf.keras.layers.SpatialDropout2D(drop)(r)
                    return r
                
                
                
                
            def res_main_block(x , filters , depth, strides = 1 , drop = 0.1):
                m = x
                for i in range(depth):
                    x = res_small_block(x , filters , strides , drop)
                    
                    if strides >1:
                        strides = 1
                m = tf.keras.layers.Conv2D(filters , (3,3) , strides = (2,2) , padding = "same")(m)
                m = tf.keras.layers.concatenate([x,m])
                m = tf.keras.layers.Conv2D(filters , (1,1) , padding = "valid")(m)
                m = tf.keras.layers.BatchNormalization()(m)
                m = Swish(m)
                m = tf.keras.layers.SpatialDropout2D(drop*2)(m)
                
                return m
                    

            #For later use of auto-scaling network by img_size
            #count = int(math.floor(math.log(img_H / small_format, 2)))
            
            
            
            
            
            inputs = tf.keras.layers.Input((img_H, img_W, channels))
            
            
            x = tf.keras.layers.Conv2D(64, (7,7) ,padding = "same")(inputs)
            
            x = res_main_block(x , 64 , 3 , strides = 2)
            x = res_main_block(x , 128 , 4 , strides = 2)
            x = res_main_block(x , 256 , 6 , strides = 2)
            x = res_main_block(x , 512 , 3 , strides = 2)
            
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            
            x = tf.keras.layers.Dense(1024)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = Swish(x)
            


                
                
                


            outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

            model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
            return model
            

        @staticmethod
        def ResNet_50(shape , n_classes):
            img_H , img_W , channels = shape
            
            def Swish(x):
                return x * tf.nn.sigmoid(x)  
            
            def conv_batch_relu(x , filters , kernel_size , strides = 1):
                x = tf.keras.layers.Conv2D(filters , kernel_size , strides , padding = "same")(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = Swish(x)
                
                return x

                    
            def identity_block(tensor , filters):
                x = conv_batch_relu(tensor , filters = filters , kernel_size = 1 , strides = 1)
                x = conv_batch_relu(x , filters = filters , kernel_size = 3 , strides = 1)
                
                x = tf.keras.layers.Conv2D(filters*4 , kernel_size = 1 , strides = 1)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Add()([tensor,x])
                x = Swish(x)
                
                return x
                
            def projection_block(tensor , filters , strides):
                x = conv_batch_relu(tensor , filters = filters , kernel_size = 1 , strides = strides)
                x = conv_batch_relu(x , filters = filters , kernel_size = 3 , strides = 1)
                x = tf.keras.layers.Conv2D(filters*4 , kernel_size = 1 , strides = 1)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                
                shortcut = tf.keras.layers.Conv2D(filters*4 , 1 , strides = strides)(tensor)
                shortcut = tf.keras.layers.BatchNormalization()(shortcut)
                
                
                x = tf.keras.layers.Add()([shortcut,x])
                x = Swish(x)
                return x
            

                

            def res_main_block(x , filters ,reps , strides):
                x = projection_block(x , filters , strides)
                
                for _ in range(reps-1):
                    x = identity_block(x , filters)
                return x


            inputs = tf.keras.layers.Input((img_H, img_W, channels))
            
            
            x = tf.keras.layers.Conv2D(64, (7,7) , strides = (2,2) ,padding = "same")(inputs)
            
            x = tf.keras.layers.MaxPool2D((2,2))(x)
            #x = tf.keras.layers.Conv2D(64*4, (1,1) , padding = "same")(x)
            
            x = res_main_block(x , 64 , 3 , strides = 1)
            x = res_main_block(x , 128 , 4 , strides = 2)
            x = res_main_block(x , 256 , 6 , strides = 2)
            x = res_main_block(x , 512 , 3 , strides = 2)
            
            x = tf.keras.layers.GlobalAveragePooling2D()(x)

            x = tf.keras.layers.Dense(256)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = Swish(x)
            
            x = tf.keras.layers.Dense(128)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = Swish(x)
            
            x = tf.keras.layers.Dense(64)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            
            x = tf.keras.layers.Dense(64)(x)
            x = tf.keras.layers.BatchNormalization()(x)
   
        
            outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
            return model 




        @staticmethod
        def ResNet_50_Dropout(shape , n_classes):
            img_H , img_W , channels = shape
            
            def Swish(x):
                return x * tf.nn.sigmoid(x)  

            
            def conv_batch_relu(x , filters , kernel_size , strides = 1):
                x = tf.keras.layers.Conv2D(filters , kernel_size , strides , padding = "same")(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = Swish(x)
                x = tf.keras.layers.SpatialDropout2D(0.05)(x)
                
                return x
       
                    
            def identity_block(tensor , filters):
                x = conv_batch_relu(tensor , filters = filters , kernel_size = 1 , strides = 1)
                x = conv_batch_relu(x , filters = filters , kernel_size = 3 , strides = 1)
                
                x = tf.keras.layers.Conv2D(filters*4 , kernel_size = 1 , strides = 1)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Add()([tensor,x])
                x = Swish(x)
                x = tf.keras.layers.SpatialDropout2D(0.05*2)(x)
                
                return x
                
            def projection_block(tensor , filters , strides):
                x = conv_batch_relu(tensor , filters = filters , kernel_size = 1 , strides = strides)
                x = conv_batch_relu(x , filters = filters , kernel_size = 3 , strides = 1)
                x = tf.keras.layers.Conv2D(filters*4 , kernel_size = 1 , strides = 1)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                
                shortcut = tf.keras.layers.Conv2D(filters*4 , 1 , strides = strides)(tensor)
                shortcut = tf.keras.layers.BatchNormalization()(shortcut)
                
                
                x = tf.keras.layers.Add()([shortcut,x])
                x = Swish(x)
                x = tf.keras.layers.SpatialDropout2D(0.05*2)(x)
                return x
            
       
                
       
            def res_main_block(x , filters ,reps , strides):
                x = projection_block(x , filters , strides)
                
                for _ in range(reps-1):
                    x = identity_block(x , filters)
                x = tf.keras.layers.SpatialDropout2D(0.05*4)(x)
                return x
       
       
            inputs = tf.keras.layers.Input((img_H, img_W, channels))
            
            
            x = tf.keras.layers.Conv2D(64, (7,7) , strides = (2,2) ,padding = "same")(inputs)
            
            x = tf.keras.layers.MaxPool2D((2,2))(x)
            #x = tf.keras.layers.Conv2D(64*4, (1,1) , padding = "same")(x)
            
            x = res_main_block(x , 64 , 3 , strides = 1)
            x = res_main_block(x , 128 , 4 , strides = 2)
            x = res_main_block(x , 256 , 6 , strides = 2)
            x = res_main_block(x , 512 , 3 , strides = 2)
            
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
       
            x = tf.keras.layers.Dense(256)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = Swish(x)
            x = tf.keras.layers.Dropout(0.05*6)(x)
            
            x = tf.keras.layers.Dense(128)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = Swish(x)
            x = tf.keras.layers.Dropout(0.05*6)(x)
            
            x = tf.keras.layers.Dense(64)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.05*4)(x)
            
            x = tf.keras.layers.Dense(64)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.05*4)(x)
       
        
            outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
            return model



        @staticmethod
        def EfficientNet_B0(shape , n_classes):
            img_H , img_W , channels = shape
            
            inputs = tf.keras.layers.Input((img_H, img_W, channels))
            
            

            
            def mb_conv_block(inputs, filter_num, expansion_factor, kernel_size, stride):

                # Expansion phase (Inverted Residual)
                x = Conv2D(filter_num*expansion_factor, kernel_size=(1, 1), padding='same', use_bias=False)(inputs) 
                x = BatchNormalization()(x)
                x = Activation('swish')(x)
                
                # Depthwise convolution phase
                x = DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same', use_bias=False)(x)
                x = BatchNormalization()(x)
                x = Activation('swish')(x)
                
                # Squeeze and Excitation phase
                se = GlobalAveragePooling2D()(x)
                se = Reshape((1, 1, filter_num*expansion_factor))(se)
                se = Conv2D(filter_num // expansion_factor, kernel_size=(1, 1), padding='same', use_bias=False)(se) 
                se = Conv2D(filter_num * expansion_factor, kernel_size=(1, 1), padding='same', use_bias=False)(se) 
                x = Multiply()([x, se])
                
                # Output phase (Linear) 
                x = Conv2D(filters=filter_num, kernel_size=(1, 1), padding='same', use_bias=False)(x)    
                x = BatchNormalization()(x)
                x = SpatialDropout2D(0.2)(x)
                # Add identity shortcut if dimensions match
                if  x.shape[-1] ==inputs.shape[-1] and stride == 1: 
                    x = Add()([x, inputs])
                
                return x
            
            def main_block(x , filter_num , expansion_factor , kernel_size , stride , depth):
                for _ in range(depth):
                    x = mb_conv_block(x, filter_num, expansion_factor, kernel_size, stride)
                    if stride >1:
                        stride = 1
                return x
                
            
            x = Conv2D(32, (3,3), strides = 2, padding = 'same')(inputs)
            
            x = main_block(x , filter_num = 16 , expansion_factor = 1 , kernel_size = 3 , stride = 1 , depth = 1)
            x = main_block(x , filter_num = 24 , expansion_factor = 6 , kernel_size = 3 , stride = 2 , depth = 2)
            x = main_block(x , filter_num = 40 , expansion_factor = 6 , kernel_size = 5 , stride = 2 , depth = 2)
            x = main_block(x , filter_num = 80 , expansion_factor = 6 , kernel_size = 3 , stride = 2 , depth = 3)
            x = main_block(x , filter_num = 112 , expansion_factor = 6 , kernel_size = 5 , stride = 1 , depth = 3)
            x = main_block(x , filter_num = 192, expansion_factor = 6 , kernel_size = 5 , stride = 2 , depth = 4)
            x = main_block(x , filter_num = 320 , expansion_factor = 6 , kernel_size = 3 , stride = 1 , depth = 1)
            
            x = Conv2D(1280 , (1,1) , strides = 1 , padding = 'same')(x)

            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            
            x = tf.keras.layers.Dense(1280)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = Activation('swish')(x)
            x = tf.keras.layers.Dropout(0.05*6)(x)
            
            x = tf.keras.layers.Dense(512)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = Activation('swish')(x)
            x = tf.keras.layers.Dropout(0.05*6)(x)
            
            x = tf.keras.layers.Dense(256)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = Activation('swish')(x)
            x = tf.keras.layers.Dropout(0.05*6)(x)
   
            
            outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
            model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
            
            return model
            
            


