#Downloading ml assets from library folder
import sys
import os
sys.path.append(os.path.dirname(__file__))
import ML_assets as ml
import Architectures as arch
#Importing rest of the libraries

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer   
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from contextlib import redirect_stdout
from matplotlib.widgets import Slider
import math
import cv2
import json
from sklearn.metrics import confusion_matrix


class Utils:
    

    def Initialize_data(DataBase_directory, Data_directory, img_H, img_W, grayscale, Load_from_CSV):
        """
        Initializes -> Loading data from main DataBase folder and load it by classes in 
        data directory

        Args:
            DataBase_directory (str):   Main DataBase directory
            Data_directory (str):       Local project resized/grayscaled DataBase directory
            img_H / img_W (int):        Image height and width in local DataBase
            grayscale (bool):           Grayscale or RGB 
            Load_from_CSV (bool):       Load data from CSV file instead of jpg/png...

        Returns:
            Database in project folder
        """    
        
        if not os.path.isdir(Data_directory):
            os.makedirs(Data_directory)
            print("Creating data storage directory...\n")
            
        if len(os.listdir(Data_directory)) == 0:
            print("There is no Dataset Initialized, initializing Dataset...")
            if Load_from_CSV:
                ml.DataSets.Create_Img_Classification_DataSet_CSV(DataBase_directory, img_H, img_W, Save_directory=Data_directory)
            else:
                ml.DataSets.Create_Img_Classification_DataSet(DataBase_directory, img_H, img_W, Save_directory=Data_directory , grayscale = grayscale)
        else:
            print("Found initialized Dataset")
            database_list = os.listdir(DataBase_directory)
            data_list = os.listdir(Data_directory)
            if Load_from_CSV:
                data_list = [element.replace(".csv" , "") for element in data_list] 
                database_list = ['x_test','x_train', 'y_test','y_train']
                
            data_list_clean = [element.replace(".npy" , "") for element in data_list]
            
            
            
            if all(elem in data_list for elem in database_list):
                print("Dataset is lacking some of the classes, initializing Dataset again")
                if Load_from_CSV:
                    ml.DataSets.Create_Img_Classification_DataSet_CSV(DataBase_directory, img_H, img_W, Save_directory=Data_directory)
                else:
                    ml.DataSets.Create_Img_Classification_DataSet(DataBase_directory, img_H, img_W, Save_directory=Data_directory , grayscale = grayscale)
            else:
                print("Dataset is initialized correctly!")
                   
    
    def Process_Data(x , y ,dataset_multiplier, DataProcessed_directory, Create_test_set, flipRotate = False , randBright = False , gaussian = False , denoise = False , contour = False ):        
        #Folder creation if not existing
        if not os.path.isdir(DataProcessed_directory):
            os.makedirs(DataProcessed_directory)
            print("Creating processed data storage directory...\n") 
        #If folder exists trying to load data from it
        else:  
            print("Found processed Dataset,loading...")
            if Create_test_set:
                try:
                    x_train = np.load(os.path.join(DataProcessed_directory ,"x_train.npy"))
                    y_train = np.load(os.path.join(DataProcessed_directory ,"y_train.npy"))
                    
                    x_val = np.load(os.path.join(DataProcessed_directory ,"x_val.npy"))
                    y_val = np.load(os.path.join(DataProcessed_directory ,"y_val.npy"))
                    
                    x_test = np.load(os.path.join(DataProcessed_directory ,"x_test.npy"))
                    y_test = np.load(os.path.join(DataProcessed_directory ,"y_test.npy"))
                    return x_train , y_train , x_val , y_val , x_test , y_test
                    
                except:
                    print("Could not load processed files, probably not present in the folder, creating...")
                
            else:
                try:
                    x_train = np.load(os.path.join(DataProcessed_directory ,"x_train.npy"))
                    y_train = np.load(os.path.join(DataProcessed_directory ,"y_train.npy"))
                    
                    x_val = np.load(os.path.join(DataProcessed_directory ,"x_val.npy"))
                    y_val = np.load(os.path.join(DataProcessed_directory ,"y_val.npy"))
                    return x_train , y_train , x_val , y_val
        
                       
                except:
                    print("Could not load processed files, probably not present in the folder, creating...")
               
    
        print("There is no Dataset processed, processing Dataset...")

        if not Create_test_set:
            x_train , x_val , y_train , y_val = train_test_split(x,y,test_size = 0.2 ,stratify = y, shuffle = True)
        else:
            x_train , x_val , y_train , y_val = train_test_split(x,y,test_size = 0.3 ,stratify = y, shuffle = True)
            x_val , x_test , y_val , y_test = train_test_split(x_val,y_val,test_size = 0.66 ,stratify = y_val, shuffle = True)
        
        print("Augmentation of images...")
        if (not (flipRotate or randBright or gaussian or denoise or contour)) and dataset_multiplier >1:
            print("\nNo augmentation specified, dataset will be just multiplied",dataset_multiplier, "times")
            
        if (not (flipRotate or randBright or gaussian or denoise or contour)) and dataset_multiplier <=1:
            print("\nNo augmentation, skipping...")
        x_train,y_train = ml.DataSets.Augment_classification_dataset(x_train, y_train, dataset_multiplier, flipRotate, randBright, gaussian, denoise, contour )            
            
        
        
        
        if Create_test_set:
            np.save(os.path.join(DataProcessed_directory ,"x_train.npy") , x_train)
            np.save(os.path.join(DataProcessed_directory ,"y_train.npy") , y_train)
            
            np.save(os.path.join(DataProcessed_directory ,"x_val.npy") , x_val)
            np.save(os.path.join(DataProcessed_directory ,"y_val.npy") , y_val)
            
            np.save(os.path.join(DataProcessed_directory ,"x_test.npy") , x_test)
            np.save(os.path.join(DataProcessed_directory ,"y_test.npy") , y_test)
            
            return x_train , y_train , x_val , y_val , x_test , y_test
            
        else:
            np.save(os.path.join(DataProcessed_directory ,"x_train.npy") , x_train)
            np.save(os.path.join(DataProcessed_directory ,"y_train.npy") , y_train)
            
            np.save(os.path.join(DataProcessed_directory ,"x_val.npy") , x_val)
            np.save(os.path.join(DataProcessed_directory ,"y_val.npy") , y_val)
            
            return x_train , y_train , x_val , y_val
    
    
    
    def Initialize_Gan_model(gan_generator, gan_discriminator, show_architecture = False):    

                
        gan_discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

        # make weights in the discriminator not trainable
        gan_discriminator.trainable = False
        # connect them
        gan_model = tf.keras.Sequential()
        # add generator
        gan_model.add(gan_generator)
        # add the discriminator
        gan_model.add(gan_discriminator)
        gan_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
        
        if show_architecture:
            gan_model.summary()
        
        return gan_model
            

    class SaveBestModel(tf.keras.callbacks.Callback):
        def __init__(self, filepath, monitor='val_loss', mode='min', min_delta=0):
            super().__init__()
            self.filepath = filepath
            self.monitor = monitor
            self.mode = mode
            self.min_delta = min_delta
            self.best_score = self._load_best_score()

            if self.best_score is None:
                self.best_score = float('inf') if mode == 'min' else float('-inf')

        def _save_best_score(self, score):
            with open(self.filepath + "_score.json", "w") as f:
                json.dump({"best_score": score}, f)

        def _load_best_score(self):
            if os.path.exists(self.filepath + "_score.json"):
                with open(self.filepath + "_score.json", "r") as f:
                    return json.load(f)["best_score"]
            else:
                return None

        def on_epoch_end(self, epoch, logs=None):
            current_score = logs.get(self.monitor)

            if current_score is not None:
                if self._is_improvement(current_score, self.best_score):
                    print(f"Epoch {epoch + 1}: Improvement detected in {self.monitor}. Saving model with score: {current_score:.4f}")
                    self.best_score = current_score
                    self.model.save(self.filepath)
                    self._save_best_score(self.best_score)


        def _is_improvement(self, current, best):
            if self.mode == 'min':
                return current < best - self.min_delta
            else:
                return current > best + self.min_delta
            
            
    def Initialize_weights_and_training(x_train, y_train, model, model_directory, train, epochs, patience, batch_size,min_delta, x_val=None, y_val=None, device = "CPU:0"):    
        #!!! Model training
        #########################################################################
        #########################################################################
        #Check if directory of trained model is present, if not, create one 
        if not os.path.isdir(model_directory):
            os.makedirs(model_directory)
            print("Creating model directory storage directory...\n")
            
        model_weights_directory = os.path.join(model_directory , "Model.keras")
        model_history_directory = os.path.join(model_directory , "Model_history.csv")
        
        model , train , starting_epoch = ml.General.Load_model_check_training_progress(model, train, epochs, model_weights_directory, model_history_directory)
        
    
             
        if train:
            #Create callback function to save best performing model
            
            if starting_epoch == 0:
                csv_append = False
            else:
                csv_append = True
                
            callbacks = [
                        #Stop if no increase in accuracy after x epochs
                        tf.keras.callbacks.EarlyStopping(patience=patience, 
                                                         monitor='val_accuracy',
                                                         min_delta=min_delta),
                        #Checkpoint model if performance is increased
                        Utils.SaveBestModel(filepath=model_weights_directory, monitor='val_accuracy', mode='max',min_delta = min_delta),
                        
                        #Save data through training
                        tf.keras.callbacks.CSVLogger(filename = model_history_directory , append = csv_append)
                        ]

            with tf.device(device):
                
                #Start measuring time
                timer_start = timer()
                model.fit(x_train,y_train,
                          initial_epoch = starting_epoch,
                          validation_data = (x_val , y_val),
                          epochs=epochs,
                          batch_size = batch_size,
                          callbacks = callbacks
                          )
                
                print("Time took to train model: ",round(timer()-timer_start),2)    
                
            
            #Save the best achieved model
            print("Loading model which was performing best during training...\n")
            model.load_weights(model_weights_directory)   
                
        
             
         
            
         
        #########################################################################
        #########################################################################
        return model
    
    

       
    def Initialize_Results(model,model_directory, dictionary,evaluate, x_train = None ,y_train = None ,x_val = None , y_val = None , x_test = None , y_test = None):    
        #!!! Model results
        #########################################################################
        #########################################################################
        
        #Plot model training history
        model_history_directory = os.path.join(model_directory , "Model_history.csv")
        Model_history = pd.read_csv(model_history_directory)
        ml.General.Model_training_history_plot_CSV(Model_history)
        
        
        try:
            #Create confusion matrix
            #Predict classes
            print("\nPredicting classes based on test set...")
            y_pred = model.predict(x_test)
            
            plt.figure()
            ml.General.Conf_matrix_classification(y_test  ,y_pred , dictionary , normalize = True)
        except:
            print("No test set provided, skipping...")
            
        try:
            #Create confusion matrix
            #Predict classes
            print("\nPredicting classes based on validation set...")
            y_pred = model.predict(x_val)
            
            plt.figure()
            ml.General.Conf_matrix_classification(y_val ,y_pred , dictionary , normalize = True)
        except:
            print("No validation set provided, skipping...")    
        
    
        if evaluate:
            try:
                #Evaluate model
                print("\nModel evaluation train set:")
                model.evaluate(x_train, y_train)
            except:
                print("No train set provided, skipping...")
                
            try:
                #Evaluate model
                print("\nModel evaluation validation set:")
                model.evaluate(x_val, y_val)
            except:
                print("No validation set provided, skipping...")
            
            try:
                #Evaluate model
                print("\nModel evaluation test set:")
                model.evaluate(x_test, y_test)
            except:
                print("No test set provided, skipping...")

class Project:
    class Classification_Project:
        def  __init__(self,config):
            #Low level constants
            self.PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(sys.argv[0]))
            #Initial
            self.DATABASE_DIRECTORY = config.Initial_params["DataBase_directory"]
            self.CREATE_TEST_SET = config.Initial_params["Create_Test_set"]
            self.CSV_LOAD = config.Initial_params["Load_from_CSV"]
            self.IMG_H = config.Initial_params["img_H"]
            self.IMG_W = config.Initial_params["img_W"]
            self.GRAYSCALE= config.Initial_params["grayscale"]
            self.DATA_TYPE = config.Initial_params["DataType"]
            
            #Augment
            self.REDUCED_SET_SIZE = config.Augment_params["reduced_set_size"]
            self.DATASET_MULTIPLIER = config.Augment_params["dataset_multiplier"]
            self.FLIPROTATE = config.Augment_params["flipRotate"]
            self.RANDBRIGHT = config.Augment_params["randBright"]
            self.GAUSSIAN = config.Augment_params["gaussian_noise"]
            self.DENOISE = config.Augment_params["denoise"]
            self.CONTOUR = config.Augment_params["contour"]
            
            #Model
            self.ARCHITECTURE_NAME = config.Model_parameters["architecture_name"]
            self.EPOCHS = config.Model_parameters["epochs"]
            self.BATCH_SIZE = config.Model_parameters["batch_size"]

            
            #High level constants
            #Form
            self.FORM = "Grayscale" if self.GRAYSCALE else "RGB"
            #Channels
            self.CHANNELS = 3 if self.FORM == "RGB" else 1
            self.CHANNELS = self.CHANNELS+1 if self.CONTOUR else self.CHANNELS
    
            cr = 0 if self.REDUCED_SET_SIZE is None else self.REDUCED_SET_SIZE
            self.PARAM_MARK = "_m"+str(self.DATASET_MULTIPLIER)+"_cr"+str(cr)+"_"+ "_".join(["1" if x else "0" for x in [self.FLIPROTATE, self.RANDBRIGHT, self.GAUSSIAN, self.DENOISE, self.CONTOUR]])
    
                    
            self.DATA_DIRECTORY = os.path.join(self.PROJECT_DIRECTORY , "DataSet" , str(str(self.IMG_H)+"x"+str(self.IMG_W)+"_"+self.FORM))
            self.DATAPROCESSED_DIRECTORY = os.path.join(self.PROJECT_DIRECTORY , "DataSet_Processed" , str(str(self.IMG_H)+"x"+str(self.IMG_W)+"_"+self.FORM),self.PARAM_MARK)
            
            
            
            
            
        def __str__(self):
            return "NO DESCRIPTION"
            
    
        def Initialize_data(self): 
            """Initializing dataset from main database folder with photos to project folder in numpy format. Photos are 
            Resized and cropped without loosing much aspect ratio, r parameter decides above what proportions of edges 
            image will be cropped to square instead of squeezed""" 
            
            Utils.Initialize_data(self.DATABASE_DIRECTORY, self.DATA_DIRECTORY, self.IMG_H, self.IMG_W, self.GRAYSCALE , self.CSV_LOAD)
            ########################################################
        def Load_and_merge_data(self):
            """Loading dataset to memory from data directory in project folder, sets can be reduced to equal size
            to eliminate disproportions if they are not same size at the main database
            In this module dictionary with names of classes is created as well, names are based on names of datsets
            Datasets names are based on the folder names in main database folder"""
            
            X, Y, DICTIONARY = ml.DataSets.Load_And_Merge_DataSet(self.DATA_DIRECTORY , self.REDUCED_SET_SIZE )
            
            return X, Y, DICTIONARY
            ########################################################
            
        def Process_data(self,X,Y):
            #3
            ########################################################
            if not self.CREATE_TEST_SET:
                X_TRAIN , Y_TRAIN, X_VAL , Y_VAL = Utils.Process_Data(X, Y, self.DATASET_MULTIPLIER, self.DATAPROCESSED_DIRECTORY, self.CREATE_TEST_SET, self.FLIPROTATE, self.RANDBRIGHT, self.GAUSSIAN, self.DENOISE, self.CONTOUR)
            
            else:
                X_TRAIN , Y_TRAIN, X_VAL , Y_VAL , X_TEST , Y_TEST = Utils.Process_Data(X, Y, self.DATASET_MULTIPLIER, self.DATAPROCESSED_DIRECTORY, self.CREATE_TEST_SET, self.FLIPROTATE, self.RANDBRIGHT, self.GAUSSIAN, self.DENOISE, self.CONTOUR)
            
            if self.CREATE_TEST_SET:
                X_TRAIN = np.array(X_TRAIN/255 , dtype = self.DATA_TYPE)
                Y_TRAIN = np.array(Y_TRAIN , dtype = self.DATA_TYPE)
                
                X_VAL = np.array(X_VAL/255 , dtype = self.DATA_TYPE)
                Y_VAL = np.array(Y_VAL , dtype = self.DATA_TYPE)
                
                X_TEST = np.array(X_TEST/255 , dtype = self.DATA_TYPE)
                Y_TEST = np.array(Y_TEST , dtype = self.DATA_TYPE)
                
                return X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, X_TEST, Y_TEST 
            
            else:
                X_TRAIN = np.array(X_TRAIN/255 , dtype = self.DATA_TYPE)
                Y_TRAIN = np.array(Y_TRAIN , dtype = self.DATA_TYPE)
                
                X_VAL = np.array(X_VAL/255 , dtype = self.DATA_TYPE)
                Y_VAL = np.array(Y_VAL , dtype = self.DATA_TYPE)
            
                return X_TRAIN, Y_TRAIN, X_VAL, Y_VAL
                
            ########################################################
            
        def Save_Data(self,x_train,y_train,x_val,y_val,dictionary = None,x_test = None,y_test = None, new_param_mark = None):
            self.X_TRAIN = x_train
            self.Y_TRAIN = y_train
            
            self.X_VAL = x_val
            self.Y_VAL = y_val
            
            
            self.X_TEST = x_test
            self.Y_TEST = y_test
            
            
            #Redefining some self variables if there is change 
            self.N_CLASSES = y_train.shape[1]
            
            if dictionary is None:
                self.DICTIONARY = [(i,"Class: "+str(i)) for i in range(self.N_CLASSES)]
            else:
                self.DICTIONARY = dictionary
            
            
            self.CLASS_OCCURENCE = []
            
            labels = np.concatenate((y_train,y_val))
            try:
                labels = np.concatenate((labels,y_test))
            except:
                pass
            
            for i in range(self.N_CLASSES):
                _, class_size = np.unique(labels[:,i],return_counts = True)
                self.CLASS_OCCURENCE.append((str(self.DICTIONARY[i][1]),int(class_size[1])))
            
            
            self.IMG_H = x_train.shape[1]
            self.IMG_W = x_train.shape[2]
            
            try:
                channels = x_train.shape[3]
            except:
                channels = 1 
            
            self.GRAYSCALE = True if channels == 1 else False
            self.FORM = "Grayscale" if self.GRAYSCALE else "RGB"
            self.DATA_TYPE = x_train.dtype
            if new_param_mark is not None:
                self.PARAM_MARK = new_param_mark
            
            
            
    
    
        def Initialize_weights_and_training(self, Model, Train = True, Patience = 10, Min_delta_to_save = 0.1, Device = "CPU", add_config_info = None):
            #5
            ########################################################
            #Create data for parameters file
            n_classes = self.N_CLASSES
            class_size = self.CLASS_OCCURENCE
            
            img_H = self.IMG_H
            img_W = self.IMG_W
            grayscale = self.GRAYSCALE
            Image_datatype = str(self.DATA_TYPE)
            Augmentation_mark = self.PARAM_MARK
            
            user_architecture_name = self.ARCHITECTURE_NAME
            Model_datatype = str("float32")
            total_params = Model.count_params()
            trainable_params, not_trainable_params = ml.General.Count_parameters(Model)
            num_layers = len(Model.layers)
            
            batch_size = self.BATCH_SIZE
            optimizer_params = str(Model.optimizer.get_config())
            loss = Model.loss
            

            
            # Create the content for the text file
            content = {
                "Number of Classes": n_classes,
                "Class Size": class_size,
                "Image Height": img_H,
                "Image Width": img_W,
                "Grayscale": grayscale,
                "Image Datatype": Image_datatype,
                "Augmentation Mark": Augmentation_mark,
                "User Architecture Name": user_architecture_name,
                "Model Datatype": Model_datatype,
                "Total Parameters": total_params,
                "Trainable Parameters": trainable_params,
                "Not-Trainable Parameters": not_trainable_params,
                "Number of Layers": num_layers,
                "Batch Size": batch_size,
                "Optimizer Parameters": optimizer_params,
                "Loss function": loss,
                "Additional information": add_config_info
            }
            
            f_name = ml.General.hash_string(str(content))
            model_directory = os.path.join("Models_saved", str(f_name))
            self.MODEL_DIRECTORY = model_directory
            
            if not os.path.exists(model_directory):
                print("Model Created")
                os.mkdir(model_directory)
            
            params_directory = os.path.join(model_directory,"Model_parameters.json")
            # Save the content to a text file
            # Write dictionary to JSON file
            with open(params_directory, 'w') as json_file:
                json.dump(content, json_file, indent=4)


            self.MODEL = Utils.Initialize_weights_and_training(x_train = self.X_TRAIN,
                                                       y_train= self.Y_TRAIN,
                                                       x_val = self.X_VAL,
                                                       y_val = self.Y_VAL,
                                                       model = Model,
                                                       model_directory = model_directory,
                                                       train = Train,
                                                       epochs = self.EPOCHS,
                                                       patience = Patience,
                                                       batch_size = self.BATCH_SIZE,
                                                       min_delta= Min_delta_to_save,
                                                       device = Device
                                                       )
            
            return self.MODEL
            ########################################################
    
        def Initialize_resulits(self,Evaluate = False):
            #6
            ########################################################
            Utils.Initialize_Results(self.MODEL,
                                  self.MODEL_DIRECTORY,
                                  self.DICTIONARY,
                                  Evaluate,
                                  self.X_TRAIN,
                                  self.Y_TRAIN,
                                  self.X_VAL,
                                  self.Y_VAL,
                                  self.X_TEST,
                                  self.Y_TEST
                                  )
            ######################################################## 
            
            
            
        def Models_analysis(self,models_directory = "Models_saved"):
            print("Collecting model Data")
            
            #Checking for folder names, filtering if something in the folder is not another folder
            unfiltered_folders = []
            for item in os.listdir(os.path.join(self.PROJECT_DIRECTORY,models_directory)):
                item_path = os.path.join(self.PROJECT_DIRECTORY,models_directory, item)
                if os.path.isdir(item_path):
                    unfiltered_folders.append(item)
                    
            #checks if model folder have necessary files, if not. Put it out of the list
            folders = []
            for item in unfiltered_folders:
                contains_all_files = True
                files = ["Model.keras", "Model.keras_score.json", "Model_history.csv", "Model_parameters.json"]
                for file in files:
                    if not os.path.isfile(os.path.join(self.PROJECT_DIRECTORY,"Models_saved",item,file)):
                        print("Model: ",item,"do not contain file: ",file,"and wont be considered in the analysis")
                        contains_all_files = False

                if contains_all_files:
                    folders.append(item)
                    
            Data_path = os.path.join(self.PROJECT_DIRECTORY,"Analysis","Data.csv")
            #Create dataframe with analysys if it does not exists yet
            if not os.path.isfile(Data_path):
                c = ["Model_ID",
                     "Model_WorkName",
                     "Model_Parameters",
                     "Model_Dtype",
                     "Batch_size",
                     "Epochs_trained",
                     "Epochs_toBest",
                     
                     "Optimizer_data",
                     "Loss_type",
                     "Metrics_type",

                     "N_classes",
                     "Img_number",
                     "Aug_mark",
                     "Img_Dtype",
                     "Form",
                     "Img_H",
                     "Img_W",
                      
                     "Loss_train",
                     "Loss_val",
                     "Loss_test",
                     "Metrics_train",
                     "Metrics_val",
                     "Metrics_test",
                     "Additional_info"
                     ]
                Data = pd.DataFrame(columns = c)
            else:
                Data = pd.read_csv(Data_path)

            #Determine if this model is already present in the data
            for model in folders:
                Model_history = pd.read_csv(os.path.join(self.PROJECT_DIRECTORY,"Models_saved",model,"Model_history.csv"))
                if Data["Model_ID"].isin([model]).any():
                    print("Model: ",model,"present in Data")
                    #To update if epochs has been changed
                else:
                    print("Collecting model data...")
                    
                    param_path = os.path.join(self.PROJECT_DIRECTORY,"Models_saved",model,"Model_parameters.json")
                    with open(param_path, 'r') as json_file:
                        param_data = json.load(json_file)

                    Model_WorkName = param_data["User Architecture Name"]
                    Model_Parameters = param_data["Total Parameters"]
                    Model_Dtype = param_data["Model Datatype"]
                    Batch_Size = param_data["Batch Size"]
                    Epochs_Trained = Model_history["epoch"].iloc[-1]
                    best_val_acc = Model_history["val_accuracy"].idxmax()
                    Epochs_toBest = (Model_history['val_accuracy'] == Model_history["val_accuracy"][best_val_acc]).idxmax()
                    
                    Optimizer_data = param_data["Optimizer Parameters"]
                    Loss_type = param_data["Loss function"]
                    Metrics_type = "to add"

                    N_classes = param_data["Number of Classes"]
                    Class_size = param_data["Class Size"]
                    Img_number = 0
                    for item in Class_size:
                        Img_number+= item[1]
                        
                        
                    Aug_mark = param_data["Augmentation Mark"]
                    Img_Dtype = param_data["Image Datatype"]
                    Form = bool(param_data["Grayscale"])
                    Form = "Grayscale" if Form else "RGB"
                    Img_H = param_data["Image Height"]
                    Img_W = param_data["Image Width"]
                    
                    processed_data_dir = os.path.join(self.PROJECT_DIRECTORY , "DataSet_Processed" , str(str(Img_H)+"x"+str(Img_W)+"_"+Form),Aug_mark)
                    
                    Keras_model = tf.keras.models.load_model(os.path.join(self.PROJECT_DIRECTORY,"Models_saved",model,"Model.keras"))
                    try:
                        x_train = (np.load(os.path.join(processed_data_dir,"x_train.npy"))/255).astype(Img_Dtype)
                        y_train = (np.load(os.path.join(processed_data_dir,"y_train.npy"))/255).astype(Img_Dtype)
                        print("Calculating train loss...")
                        Loss_train,_ = Keras_model.evaluate(x_train,y_train , verbose = 1)
                    except:
                        print("No train set found")
                        Loss_train = None

                    try:
                        x_val = (np.load(os.path.join(processed_data_dir,"x_val.npy"))/255).astype(Img_Dtype)
                        y_val = (np.load(os.path.join(processed_data_dir,"y_val.npy"))/255).astype(Img_Dtype)
                        print("Calculating validation loss...")
                        Loss_val,_ = Keras_model.evaluate(x_val,y_val, verbose = 1)
                    except:
                        print("No validation set found")
                        Loss_val = None
                        
                    try:
                        x_test = (np.load(os.path.join(processed_data_dir,"x_test.npy"))/255).astype(Img_Dtype)
                        y_test = (np.load(os.path.join(processed_data_dir,"y_test.npy"))/255).astype(Img_Dtype)
                        print("Calculating test loss...")
                        Loss_test,_ = Keras_model.evaluate(x_test,y_test,verbose = 1)
                    except:
                        print("No test set found")
                        Loss_test = None 
                          
                    
                    Additional_info = param_data["Additional information"]


                    Data.loc[model] = [ model,
                                        Model_WorkName,
                                        Model_Parameters,
                                        Model_Dtype,
                                        Batch_Size,
                                        Epochs_Trained,
                                        Epochs_toBest,
                                     
                                        Optimizer_data,
                                        Loss_type,
                                        Metrics_type,
            
                                        N_classes,
                                        Img_number,
                                        Aug_mark,
                                        Img_Dtype,
                                        Form,
                                        Img_H,
                                        Img_W,
                                      
                                        Loss_train,
                                        Loss_val,
                                        Loss_test,
                                        "Metrics_train [TP,FP,FN,TN]",
                                        "Metrics_val [TP,FP,FN,TN]",
                                        "Metrics_test [TP,FP,FN,TN]",
                                        Additional_info
                                      ]
                    Data.to_csv(Data_path)
            
            #At the end of Data creation
            Data.to_csv(Data_path,index = False)

                

            
            
        
            
        def Generate_sample_submission(self, filepath = None):
            if filepath is None:
                sample_submission = pd.read_csv(os.path.join(self.DATA_DIRECTORY , "sample_submission.csv")) 

            #img_id = sample_submission.columns[0]
            label = sample_submission.columns[1]
            try:
                label_array = np.argmax(self.MODEL.predict(self.X_TEST), axis = 1)
            except:
                label_array = self.Y_TEST
            sample_submission[label] = label_array
            return sample_submission
    




    
    class Gan_Project:
        def  __init__(self,config):
            #Low level constants
            self.PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(sys.argv[0]))
            #Initial
            self.DATABASE_DIRECTORY = config.Initial_params["DataBase_directory"]
            self.KAGGLE_SET = config.Initial_params["Kaggle_set"]
            self.CSV_LOAD = config.Initial_params["Load_from_CSV"]
            self.IMG_H = config.Initial_params["img_H"]
            self.IMG_W = config.Initial_params["img_W"]
            self.GRAYSCALE= config.Initial_params["grayscale"]
            self.DATA_TYPE = config.Initial_params["DataType"]
            
            #Augment
            self.REDUCED_SET_SIZE = config.Augment_params["reduced_set_size"]
            self.DATASET_MULTIPLIER = config.Augment_params["dataset_multiplier"]
            self.FLIPROTATE = config.Augment_params["flipRotate"]
            self.RANDBRIGHT = config.Augment_params["randBright"]
            self.GAUSSIAN = config.Augment_params["gaussian_noise"]
            self.DENOISE = config.Augment_params["denoise"]
            self.CONTOUR = config.Augment_params["contour"]
            
            #Model
            self.GENERATOR_ARCHITECTURE = config.Model_parameters["generator_architecture"]
            self.DISCRIMINATOR_ARCHITECTURE = config.Model_parameters["discriminator_architecture"]
            self.SHOW_ARCHITECTURE = config.Model_parameters["show_architecture"]
            self.DEVICE = config.Model_parameters["device"]
            self.TRAIN = config.Model_parameters["train"]
            self.EPOCHS = config.Model_parameters["epochs"]
            #self.PATIENCE = config.Model_parameters["patience"]
            self.LATENT_DIM = config.Model_parameters["latent_dim"]
            self.BATCH_SIZE = config.Model_parameters["batch_size"]
            self.SAMPLE_INTERVAL = config.Model_parameters["sample_interval"]
            self.SAMPLE_NUMBER = config.Model_parameters["sample_number"]
            #self.MIN_DELTA = config.Model_parameters["min_delta"]
            self.EVALUATE = config.Model_parameters["evaluate"]
            
            #High level constants
            #Form
            self.FORM = "Grayscale" if self.GRAYSCALE else "RGB"
            #Channels
            self.CHANNELS = 3 if self.FORM == "RGB" else 1
            self.CHANNELS = self.CHANNELS+1 if self.CONTOUR else self.CHANNELS
            
            cr = 0 if self.REDUCED_SET_SIZE is None else self.REDUCED_SET_SIZE
            self.PARAM_MARK = "_m"+str(self.DATASET_MULTIPLIER)+"_cr"+str(cr)+"_"+ "_".join(["1" if x else "0" for x in [self.FLIPROTATE, self.RANDBRIGHT, self.GAUSSIAN, self.DENOISE, self.CONTOUR]])
    
                    
            self.DATA_DIRECTORY = os.path.join(self.PROJECT_DIRECTORY , "DataSet" , str(str(self.IMG_H)+"x"+str(self.IMG_W)+"_"+self.FORM))
            self.DATAPROCESSED_DIRECTORY = os.path.join(self.PROJECT_DIRECTORY , "DataSet_Processed" , str(str(self.IMG_H)+"x"+str(self.IMG_W)+"_"+self.FORM),self.PARAM_MARK)
            
            self.MODEL_ARCHITECTURE = ''.join([self.GENERATOR_ARCHITECTURE , "__" , self.DISCRIMINATOR_ARCHITECTURE])
            self.MODEL_DIRECTORY =  os.path.join(self.PROJECT_DIRECTORY , "Models_saved" , self.MODEL_ARCHITECTURE , self.FORM , str(str(self.IMG_H)+"x"+str(self.IMG_W)) , str("bs"+str(self.BATCH_SIZE) + self.PARAM_MARK)  )
            self.MODEL_NAME = str(self.MODEL_ARCHITECTURE + "_bs"+str(self.BATCH_SIZE)+".keras")
            
            self.MODEL_WEIGHTS_DIRECTORY = os.path.join(self.MODEL_DIRECTORY , self.MODEL_NAME)
            self.MODEL_HISTORY_DIRECTORY = os.path.join(self.MODEL_DIRECTORY , "Model_history.csv")
        ########################################################    
            
        def __str__(self):
            return f"This is class representing the project, main parameters are:\n\nOriginalDatabase: {self.DATABASE_DIRECTORY}\nGenerator Used: {self.GENERATOR_ARCHITECTURE}\nDiscriminator Used: {self.DISCRIMINATOR_ARCHITECTURE}"
            
        ########################################################
    
        def Initialize_data(self): 
            """Initializing dataset from main database folder with photos to project folder in numpy format. Photos are 
            Resized and cropped without loosing much aspect ratio, r parameter decides above what proportions of edges 
            image will be cropped to square instead of squeezed""" 
            
            Utils.Initialize_data(DataBase_directory = self.DATABASE_DIRECTORY, 
                                  Data_directory = self.DATA_DIRECTORY, 
                                  img_H = self.IMG_H, 
                                  img_W = self.IMG_W, 
                                  grayscale = self.GRAYSCALE, 
                                  Load_from_CSV = self.CSV_LOAD
                                  )
            
        ########################################################
        
        def Load_and_merge_data(self):
            """Loading dataset to memory from data directory in project folder, sets can be reduced to equal size
            to eliminate disproportions if they are not same size at the main database
            In this module dictionary with names of classes is created as well, names are based on names of datsets
            Datasets names are based on the folder names in main database folder"""
            if not self.KAGGLE_SET:
                self.X_TRAIN, self.Y_TRAIN, self.DICTIONARY = ml.DataSets.Load_And_Merge_DataSet(self.DATA_DIRECTORY , self.REDUCED_SET_SIZE )
                self.N_CLASSES = len(self.DICTIONARY)
                
            else:
                #To add searching for key words such as test, x, train etc. as for now just name csvs like train, test
                self.X_TRAIN = np.load(os.path.join(self.DATA_DIRECTORY , "x_train.npy"))
                self.Y_TRAIN = np.load(os.path.join(self.DATA_DIRECTORY , "y_train.npy"))
                
                self.X_TEST = np.load(os.path.join(self.DATA_DIRECTORY , "x_test.npy"))
                try:
                    self.Y_TEST = np.load(os.path.join(self.DATA_DIRECTORY , "y_test.npy"))
                except:
                    self.Y_TEST = np.load(os.path.join(self.DATA_DIRECTORY , "y_test.npy"),allow_pickle = True)
                
        ########################################################
            
        def Process_data(self):

            if self.KAGGLE_SET:
                self.X_TRAIN , self.Y_TRAIN, self.X_VAL , self.Y_VAL = Utils.Process_Data(x = self.X_TRAIN,
                                                                                          y = self.Y_TRAIN,
                                                                                          dataset_multiplier = self.DATASET_MULTIPLIER,
                                                                                          DataProcessed_directory = self.DATAPROCESSED_DIRECTORY,
                                                                                          Kaggle_set = self.KAGGLE_SET,
                                                                                          flipRotate = self.FLIPROTATE,
                                                                                          randBright = self.RANDBRIGHT,
                                                                                          gaussian = self.GAUSSIAN,
                                                                                          denoise = self.DENOISE,
                                                                                          contour = self.CONTOUR
                                                                                          )

            else:
                self.X_TRAIN , self.Y_TRAIN, self.X_VAL , self.Y_VAL , self.X_TEST , self.Y_TEST = Utils.Process_Data(x = self.X_TRAIN,
                                                                                                                      y = self.Y_TRAIN,
                                                                                                                      dataset_multiplier = self.DATASET_MULTIPLIER,
                                                                                                                      DataProcessed_directory = self.DATAPROCESSED_DIRECTORY,
                                                                                                                      Kaggle_set = self.KAGGLE_SET,
                                                                                                                      flipRotate = self.FLIPROTATE,
                                                                                                                      randBright = self.RANDBRIGHT,
                                                                                                                      gaussian = self.GAUSSIAN,
                                                                                                                      denoise = self.DENOISE,
                                                                                                                      contour = self.CONTOUR
                                                                                                                      )
            
            try:
                self.X_TRAIN = np.array((self.X_TRAIN/255-0.5)*2 , dtype = self.DATA_TYPE)
                self.Y_TRAIN = np.array(self.Y_TRAIN , dtype = self.DATA_TYPE)
                
                self.X_VAL = np.array((self.X_VAL/255-0.5)*2 , dtype = self.DATA_TYPE)
                self.Y_VAL = np.array(self.Y_VAL , dtype = self.DATA_TYPE)
                
                self.X_TEST = np.array((self.X_TEST/255-0.5)*2 , dtype = self.DATA_TYPE)
                self.Y_TEST = np.array(self.Y_TEST , dtype = self.DATA_TYPE)
            except Exception as e:
                print("Could not standarize data:",e)
                
            
        ########################################################
        
            
        def Initialize_model_from_library(self):
            
            g_arch = f"{self.GENERATOR_ARCHITECTURE}"
            d_arch = f"{self.DISCRIMINATOR_ARCHITECTURE}"
            generator_class = getattr(arch.Gan, g_arch, None)
            discriminator_class = getattr(arch.Gan, d_arch, None)
            
            if (generator_class and discriminator_class) is not None:
                self.GENERATOR = generator_class(self.LATENT_DIM)
                self.DISCRIMINATOR = discriminator_class()
                print("Found generator named: ",g_arch,"\nFound discriminator named: ",d_arch)
            else:
                if generator_class is None:
                    print("Could not find generator class named: ",g_arch)
                    return
                if discriminator_class is None:
                    print("Could not find discriminator class named: ",d_arch)
                    return
                
            self.MODEL = Utils.Initialize_Gan_model(gan_generator = self.GENERATOR,
                                                    gan_discriminator = self.DISCRIMINATOR,
                                                    show_architecture = self.SHOW_ARCHITECTURE
                                                    )
            
            
 
        def Callback(self,current_epoch, constant_noise):
            #1    
            #Saving npy images   
            if current_epoch % self.SAMPLE_INTERVAL == 0:
                if not os.path.isdir(os.path.join(self.MODEL_DIRECTORY , "Images")):
                    os.makedirs(os.path.join(self.MODEL_DIRECTORY , "Images"))
                    print("\nCreating model directory storage directory...")
                    
                print('\nSaving',self.SAMPLE_NUMBER,'samples from',current_epoch,'epoch')
                filename = os.path.join(self.MODEL_DIRECTORY, 'Images', 'Epoch_%03d.npy' %current_epoch )
                checkpoint_samples = self.GENERATOR.predict(constant_noise)
                if len(checkpoint_samples.shape) == 4 and checkpoint_samples.shape[3]==1:
                    checkpoint_samples = np.squeeze(checkpoint_samples, axis = -1)
                # create 'fake' class labels (0)
                if len(checkpoint_samples) < self.SAMPLE_NUMBER:
                    for i in range(self.SAMPLE_NUMBER // len(checkpoint_samples) +1):
                        value = math.log(i+1.6)
                        temp = self.GENERATOR.predict(constant_noise*value)
                        if len(temp.shape) == 4 and temp.shape[3]==1:
                            temp = np.squeeze(temp, axis = -1)
                        checkpoint_samples = np.vstack((checkpoint_samples , temp ))
                        
                checkpoint_samples = (checkpoint_samples[0:self.SAMPLE_NUMBER]+1)/2  
                checkpoint_samples = np.array(checkpoint_samples*255 , dtype = np.uint8)
                np.save(filename, checkpoint_samples)   
                   
            """
            #2
            #Saving model
            if val_acc> max_vall_acc:
                if val_acc-max_vall_acc>=delta:
                    save_model
                else:
                    counter+=1
            if counter==patience:
                stop_training
            """
            #3
            #Saving history
            if self.csv_append:
                model_history = pd.read_csv(self.MODEL_HISTORY_DIRECTORY)
                
                next_index = len(model_history)  
                model_history.loc[next_index, 'epoch'] = current_epoch
                
                model_history.to_csv(self.MODEL_HISTORY_DIRECTORY, index = False)
            else:
                c = ["epoch"]
                model_history = pd.DataFrame(columns = c)
                
                next_index = len(model_history)  
                model_history.loc[next_index, 'epoch'] = int(current_epoch)
                model_history.to_csv(self.MODEL_HISTORY_DIRECTORY, index = False)
                
                self.csv_append = True
                
            self.MODEL.save(self.MODEL_WEIGHTS_DIRECTORY)   
                
                
            ########################################################
    
        def Initialize_weights_and_training_gan(self, precompiled_model=None, precompiled_generator = None, precompiled_discriminator = None):
            #5
            ########################################################
            if precompiled_model and precompiled_generator and precompiled_discriminator:
                # Use the provided precompiled model
                self.MODEL = precompiled_model
                self.GENERATOR = precompiled_generator
                self.DISCRIMINATOR = precompiled_discriminator
            else:
                # Use the initialized model from Initialize_model function
                assert hasattr(self, 'MODEL'), "Model not initialized. Call Initialize_model_from_library first or use custom compiled model, f.e, from keras or your own."
            

            #Check if directory of trained model is present, if not, create one 
            if not os.path.isdir(self.MODEL_DIRECTORY):
                os.makedirs(self.MODEL_DIRECTORY)
                print("Creating model directory storage directory...\n")

            
            self.MODEL , train , starting_epoch = ml.General.Load_model_check_training_progress(model = self.MODEL,
                                                                                                train = self.TRAIN,
                                                                                                epochs_to_train = self.EPOCHS,
                                                                                                model_weights_directory = self.MODEL_WEIGHTS_DIRECTORY,
                                                                                                model_history_directory = self.MODEL_HISTORY_DIRECTORY
                                                                                                )
            try:
                starting_epoch = int(starting_epoch)
            except:
                pass
                #No starting epoch, or its NONE
                 
            if train:
                #Create callback function to save best performing model
                
                if starting_epoch == 0:
                    self.csv_append = False
                else:
                    self.csv_append = True
                try:
                    print("Loading Constant noise...")
                    self.CONSTANT_NOISE = np.load(os.path.join(self.MODEL_DIRECTORY , "Constant_noise.npy"))
                    
                except:
                    print("Failed to load constant noise, generating one...")
                    constant_noise = np.random.normal(0, 1, (self.SAMPLE_NUMBER, self.LATENT_DIM))
                    np.save(os.path.join(self.MODEL_DIRECTORY, "Constant_noise.npy") , constant_noise)
                    self.CONSTANT_NOISE = constant_noise
                    
                
                #Deleting images if training from scratch
                if os.path.isdir(os.path.join(self.MODEL_DIRECTORY , "Images")) and starting_epoch ==0:
                    delete_imgs = True
                else:
                    delete_imgs = False
                
                timer_start = timer()
                #To add stable noise over continued training, now its only during one session
                with tf.device(self.DEVICE):
                    for epoch in range(starting_epoch+1 , self.EPOCHS+1):
                        print("\nEpoch:",epoch)
                        steps_per_epoch = len(self.X_TRAIN) // self.BATCH_SIZE
                        for step in tqdm(range(steps_per_epoch)):
                            with redirect_stdout(open(os.devnull, 'w')):
                                #1
                                #Taking batch of real samples from dataset
                                x_real, y_real = ml.General.generate_real_samples(self.X_TRAIN, self.BATCH_SIZE//2)
                                
                                #2
                                #Generating batch of fake samples from generator
                                x_fake , y_fake = ml.General.generate_fake_samples(self.GENERATOR, self.LATENT_DIM, self.BATCH_SIZE//2)
                                
                                #3
                                #Preparing combined real-fake set for discriminator to train
                                x = np.vstack((x_real,x_fake))
                                y = np.vstack((y_real, y_fake))
                                
                                #4
                                #Training discriminator
                                discriminator_loss = self.DISCRIMINATOR.train_on_batch(x,y)
                                
                                #5
                                #Update generator via discriminator error
                                noise = np.random.normal(0, 1, (self.BATCH_SIZE, self.LATENT_DIM))
                                ones = np.ones((self.BATCH_SIZE, 1))
                                generator_loss = self.MODEL.train_on_batch(noise, ones)
                                
                                
                        # Print the progress
                        sys.stdout.write(f"\n[D loss: {discriminator_loss[0]:.3f} | D acc: {discriminator_loss[1]:.3f}] [G loss: {generator_loss:.3f}]")    
                        
                        #Delete images if first iteration
                        if delete_imgs:
                            print("Deleting remaining images from folder 'Images'... ")
                            ml.General.delete_files_in_folder(os.path.join(self.MODEL_DIRECTORY , "Images"))
                            delete_imgs = False
                            
                        
                        # Save generated images every sample_interval
                        #gan_callback()
                        self.Callback(current_epoch = epoch,
                                      constant_noise = self.CONSTANT_NOISE
                                      )
                            
                        #To make in callbacks some kind of stable random noise to have nice animation of training
                    
                    
                    print("Time took to train model: ",round(timer()-timer_start),2)    
                    
                
                #Save the best achieved model
                print("Loading model which was performing best during training...\n")
                self.MODEL.load_weights(self.MODEL_WEIGHTS_DIRECTORY)   



                                

        def Initialize_history(self,show_plot, plot_size , create_gif = False, gif_scale = 1, gif_fps = 20):
            if plot_size**2>self.SAMPLE_NUMBER:
                print("Not enough samples, consider reducing plot size")
                return
            #1
            #Loading most actual model to initialize results
            try:
                print("Trying to load most actual model...")
                self.MODEL.load_weights(self.MODEL_WEIGHTS_DIRECTORY)  
            except:
                print("Could not load most actual model, working with current one loaded")
            
            #2
            #Creating 
            history_array = []
            path = os.path.join(self.MODEL_DIRECTORY , "Images")
            img_list = os.listdir(path)
            
            for filename in img_list:
                sample = np.load(os.path.join(path,filename))
                history_array.append(sample)
                
            history_array = np.array(history_array)
            
            if len(history_array) == 0:
                print("There is no data, try to train model a little first")
                return
            
            if create_gif:
                grid_array = ml.General.create_image_grid(history_array , size = plot_size)
                ml.General.create_gif(gif_array = grid_array,
                                      gif_filepath = os.path.join(self.MODEL_DIRECTORY , "Training_history.gif"),
                                      gif_height = int(grid_array.shape[1]*gif_scale),
                                      gif_width = int(grid_array.shape[2]*gif_scale),
                                      fps = gif_fps
                                      )
            
            
            
            def update_plot(epoch):
                ax.clear()  # Clear the previous image
                plt.title("Training history\nEpoch: "+str(epoch))
                if self.GRAYSCALE:
                    ax.imshow(grid_array[epoch], cmap="gray")
                else:
                    ax.imshow(grid_array[epoch])
                    
                # Optionally, update the title
                ax.axis("off")
                plt.draw()
  
            if show_plot:  
                #plt.subplots_adjust(bottom=0.25)
                # Create the figure and the axis
                fig, ax = plt.subplots()
                plt.subplots_adjust(left=0.25, bottom=0.25)
                
                ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
                slider = Slider(ax_slider, 'Epoch', 0, len(history_array)-1, valinit=0, valstep=1)
                
                # Update the plot when the slider value changes
                slider.on_changed(update_plot)
                
                # Initialize the first plot
            
                update_plot(0)
                
                plt.show()
            else:
                slider = None
                
            
            return history_array , slider
            
        
        def Initialize_results(self,show_plot, plot_size = 4, saveplot = False, plot_scale = 1):
            #1
            #Loading most actual model to initialize results
            try:
                print("Trying to load most actual model...")
                self.MODEL.load_weights(self.MODEL_WEIGHTS_DIRECTORY)  
            except:
                print("Could not load most actual model, working with current one loaded")
            
            Gen_imgs , _ = ml.General.generate_fake_samples(gan_generator = self.GENERATOR,
                                                    latent_dim = self.LATENT_DIM,
                                                    n_samples = plot_size**2
                                                    )
            Gen_imgs = (Gen_imgs+1)/2
            Gen_imgs = np.array(Gen_imgs*255 , dtype = np.uint8)
            
            Grid_img = ml.General.create_image_grid(Gen_imgs, plot_size)

        
        
            if show_plot:
                plt.figure()
                plt.axis("off")
                plt.title("Model Results")
                if self.GRAYSCALE:
                    plt.imshow(Grid_img , cmap = 'gray')
                else:
                    plt.imshow(Grid_img)
            if saveplot:
                save_dir = os.path.join(self.MODEL_DIRECTORY , 'Generator_results.png')
                Resized_grid_img = cv2.resize(Grid_img, (int(Grid_img.shape[0]*plot_scale), int(Grid_img.shape[1]*plot_scale)), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(save_dir,Resized_grid_img)
                
            return Gen_imgs
                

                    
            
        def Initialize_results_interpolation(self,n_variations, steps_to_variation, create_gif = False, gif_scale = 1,gif_fps = 20):
            gen_img_list = []
            n_vectors = n_variations
            steps = steps_to_variation
            #Interpolated latent vectors for smooth transition effect
            latent_vectors = [np.random.randn(self.LATENT_DIM) for _ in range(n_vectors-1)]
            latent_vectors.append(latent_vectors[0])
            interpolated_latent_vectors = []
            for i in range(len(latent_vectors)-1):
                for alpha in np.linspace(0, 1, steps, endpoint=False):
                    interpolated_vector = latent_vectors[i] * (1 - alpha) + latent_vectors[i + 1] * alpha
                    interpolated_latent_vectors.append(interpolated_vector)
            # Add the last vector to complete the sequence

            for vector in tqdm(interpolated_latent_vectors,desc = "Creating interpolation plot..."):
                r_vector = np.reshape(vector , (1,len(vector)))
                
                gen_img = self.GENERATOR.predict(r_vector , verbose = 0)
                if len(gen_img.shape) >= 4 and not self.GRAYSCALE:
                    gen_img = np.reshape(gen_img,(self.IMG_H,self.IMG_W,3))
                    
                if len(gen_img.shape) >= 3 and self.GRAYSCALE:
                    gen_img = np.reshape(gen_img,(self.IMG_H,self.IMG_W))
                
                gen_img = (gen_img+1)/2
                gen_img_list.append(gen_img)
                ##########


                
                
            gen_img_list = np.array(gen_img_list)
            if create_gif:
                try:
                    ml.General.create_gif(gif_array = (gen_img_list*255).astype(np.uint8),
                                          gif_filepath = os.path.join(self.MODEL_DIRECTORY , "Model_interpolation.gif"),
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
                if self.GRAYSCALE:
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
            
            
            
                        
                    
            
            
            
            
            
            
            
            
            
            
            
            
            
            

            
