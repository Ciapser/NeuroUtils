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
                    y = General.OneHot_decode(labels)
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
          
       
        
            
    def Load_And_Merge_DataSet(Data_directory , samples_per_class = None):

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
            
            
              

    def Augment_classification_dataset(x, y, dataset_multiplier, flipRotate = False , randBright = False , gaussian = False , denoise = False , contour = False ):
        n_classes = y.shape[1]
        lenght = x.shape[0]
        img_H = x.shape[1]
        img_W = x.shape[2]
        try:
            channels = x.shape[3]
        except:
            channels = 1
        
        blank_class_y = np.zeros((0,n_classes) , dtype = np.uint8)
        if channels == 1:
            blank_class_x = np.zeros((0,img_H,img_W) , dtype = np.uint8)
        else:
            blank_class_x = np.zeros((0,img_H,img_W,channels) , dtype = np.uint8)
            
        if dataset_multiplier == 1 :
            pass
            print("No augmentation specified, loading only original images")
        else:
            print("Augmenting dataset:")
            for p in range(dataset_multiplier - 1):
            
                x_aug = []
                for i in tqdm(range(lenght)):
                    if np.mean(x[i]) == 0:
                        x[i,0,0,0] = 1
                    aug_img = ImageProcessing.augment_image(x[i] , flipRotate , randBright , gaussian , denoise ,  contour)
        
                    x_aug.append(aug_img)
                    
                x_aug = np.asarray(x_aug)
                
                blank_class_x = np.concatenate( (blank_class_x , x_aug) )
                
                blank_class_y = np.concatenate( (blank_class_y , y) )
            
            #If contour is True
            if contour:
                x_aug = []
                for i in tqdm(range(lenght)):
                    aug_img = ImageProcessing.contour_mod(x[i] , density = 2)
                    x_aug.append(aug_img)
                x = np.asarray(x_aug)
               
                
        
        
        
            x = np.concatenate((x, blank_class_x))
            y = np.concatenate((y, blank_class_y))
        
        return x , y
    
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
        
    #Combined function of augmentation to use
    def augment_image(image , rand_bright = False , gaussian = False , denoise = False , flip_rotate = False , contour = False   ):
        
        if rand_bright:
            image = ImageProcessing.random_brightness(image)
        if gaussian:
            image = ImageProcessing.add_gaussian_noise(image , 0.3)
        if denoise:
            image = ImageProcessing.denoise_img(image)
        if flip_rotate:
            image = ImageProcessing.random_rotate_flip(image)
        if contour:
            image = ImageProcessing.contour_mod(image)
            
        return image        

class General:
    
    def Load_model_check_training_progress(model , train , epochs_to_train, model_weights_directory, model_history_directory):
        starting_epoch = None
        if train:
            try:
                Model_history = pd.read_csv(model_history_directory)
                starting_epoch = Model_history["epoch"].iloc[-1]
                try:
                    best_val_acc = Model_history["val_accuracy"].idxmax()
                    
                    best_val_loss = round(Model_history["val_loss"][best_val_acc],3)
                    best_val_acc = round(Model_history["val_accuracy"][best_val_acc],3)
                except:
                    print("Could not load model scores...")
        
                print("Found existing model trained for ",starting_epoch," epochs")
                try:
                    print("Best model score aqcuired in ",starting_epoch," epoch\nVal_acc: ",best_val_acc,"\nVal_loss: ",best_val_loss,)
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
        if len(x.shape) == 4:
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
    
    
    
    