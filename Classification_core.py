#Downloading my assets from library folder
import ML_assets as ml
from ML_assets import General

#Downloading config
import config

#Importing rest of the libraries
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer   
import pandas as pd
from tqdm import tqdm
import sklearn as skl





def main():
    #Checking Tensorflow Version and available computing devices
    print('Tensorflow version: ' , tf.__version__,"\n\n-------------------------------\n")
    print('Detected Devices: \n' , tf.config.list_physical_devices(),"\n-------------------------------\n")
    
    
    #!!!#Setting up parameters
    #########################################################################
    #########################################################################
    
    #Initial parameters
    DataBase_directory = config.Initial_params["DataBase_directory"]
    Kaggle_competition_dataset = config.Initial_params["Kaggle_competition_dataset"]
    Stratification_test = config.Initial_params["Stratification_test"]
    grayscale = config.Initial_params["grayscale"]
    img_H = config.Initial_params["img_H"]
    img_W = config.Initial_params["img_W"]
    DataType = config.Initial_params["DataType"]
    
    
    #Augument parameters
    reduced_set_size = config.Augment_params["reduced_set_size"]
    train_dataset_multiplier = config.Augment_params["train_dataset_multiplier"]
    
    
    parameters = [("flipRotate" ,   config.Augment_params["flipRotate"]) ,
                  ("randBright" ,   config.Augment_params["randBright"]) ,
                  ("gaussian" ,     config.Augment_params["gaussian_noise"]) ,
                  ("denoise" ,      config.Augment_params["denoise"]) ,
                  ("contour" ,      config.Augment_params["contour"])]
    
    
    #Model parameters
    model_architecture = config.Model_parameters["model_architecture"]
    device = config.Model_parameters["device"]
    train = config.Model_parameters["train"]  #Train model or only load results to plots, evaluations etc.
    epochs = config.Model_parameters["epochs"]
    patience = config.Model_parameters["patience"]
    batch_size = config.Model_parameters["batch_size"]
    evaluate = config.Model_parameters["batch_size"]
    show_architecture = config.Model_parameters["show_architecture"]
    
    #Other parameters
    if Kaggle_competition_dataset:
        Competition_dictionary = config.Other_parameters["Competition_dictionary"]
    #########################################################################
    #########################################################################
    
    
    
    
    
    #!!! Parameters to variables calculation
    #########################################################################
    #########################################################################
    str_parameters = str("_reducet_set_size"+str(reduced_set_size)  + "_multiplier"+ str(train_dataset_multiplier))
    for item in parameters:
        if item[1]:
            str_parameters += str("_"+item[0])
        del item
    
    
    #Dependent parameters
    if grayscale:
        channels = 1
    else:
        channels = 3
    
    #If contour is True
    if parameters[4][1]:
        channels+=1
        
    #########################################################################
    #########################################################################    
        
        
    
    #!!!Data loading
    #########################################################################
    #########################################################################
    if grayscale:
        form = "Grayscale"
    else:
        form = "RGB"
    
    
    Data_toRun_directory = os.path.join(os.path.dirname(os.getcwd()) , "DataSet_toRun" , str(str(img_H)+"x"+str(img_W)+"_"+form+str_parameters))
    #Create Data_path
    Data_directory = os.path.join(os.path.dirname(os.getcwd()) , "DataSet" , str(str(img_H)+"x"+str(img_W)+"_"+form))
    
    
    if not os.path.isdir(Data_toRun_directory):
        os.makedirs(Data_toRun_directory)
        print("Creating augmented data storage directory...\n")
    
    try:
        print("Trying to load augmented images...\n")
        x_train = np.load(os.path.join(Data_toRun_directory , "x_train.npy"))
        y_train = np.load(os.path.join(Data_toRun_directory , "y_train.npy"))
        
        x_val = np.load(os.path.join(Data_toRun_directory , "x_val.npy"))
        y_val = np.load(os.path.join(Data_toRun_directory , "y_val.npy"))
        if not Kaggle_competition_dataset:
            x_test = np.load(os.path.join(Data_toRun_directory , "x_test.npy"))
            y_test = np.load(os.path.join(Data_toRun_directory , "y_test.npy"))
        
        with open(os.path.join(Data_toRun_directory , "dictionary.json"), 'r') as json_file:
            dictionary = json.load(json_file)
            
        
    
        
        print("Augmented dataset loaded, ready to work with Neural Network!\n")
        
    except:
        print("No dataset of given parameters found, or data is corrupted/incomplete,    Creating new augmented dataset...\n")
        
        #If data not exists then create it with directory
        if not os.path.isdir(Data_directory):
            print("No numpy data exists in project, creating numpy data of given resolution from main database folder...\n")
            print("Note:  This data exists to minimize time of computation while working on project, can be deleted from folder when archiwized, script will create it again if necessary\n")
            #create directory if it does not exists
            os.makedirs(Data_directory)
            #Create dataset and dictionary from photo database
            ml.DataSets.Create_Img_Classification_DataSet(DataBase_directory, img_H, img_W, Save_directory=Data_directory , grayscale = grayscale)
           
        if not Kaggle_competition_dataset:    
            #Data in folder is saved to reduce computation over and over,
            #Data is merged and its amount is reduced if specified
            x_train , y_train , dictionary = ml.DataSets.Load_And_Merge_DataSet(Data_directory , samples_per_class = reduced_set_size)
            
            #Saved in less memory hungry variable, then divided by 255 to operate well in neural network
            #x_train = x_train.astype(np.float32)
            #y_train = y_train.astype(np.float16)
            
            #x_train /= 255
            
            #Split data into train, validation and test subsets, important to do it BEFORE augmentation
            x_train, x_val, y_train, y_val = train_test_split(x_train , y_train,stratify = y_train , test_size=0.3)
            x_val, x_test, y_val, y_test = train_test_split(x_val , y_val,stratify = y_val , test_size=0.66)
            
        else:
            x ,_,_ = ml.DataSets.Load_And_Merge_DataSet(Data_directory , samples_per_class = reduced_set_size)
            #Split data into train, validation and test subsets, important to do it BEFORE augmentation
            #Read files
            train_id = pd.read_csv( os.path.join(DataBase_directory , "train.csv"))
            test_id = pd.read_csv( os.path.join(DataBase_directory , "test.csv"))
            
            test_id = test_id["Image"].tolist()
            test_id = [x - 1 for x in test_id]
            x_test = x[test_id]
            
            train_labels = train_id["Mushroom"].tolist()
            train_id = train_id["Image"].tolist()
            train_id = [x -1 for x in train_id]
            x_train = x[train_id]
            
            y_train = ml.General.OneHot_decode(train_labels)
            

            dictionary = Competition_dictionary
            
            x_train, x_val, y_train, y_val = train_test_split(x_train , y_train,stratify = y_train , test_size=0.15)
            del x
            del train_id
            del train_labels
            del test_id
            
        #Test of Stratification correctness
        if Stratification_test:
            print("Newly created dataset stratification test: \n")
            for i in range(len(dictionary)):
                print("Class",i,"share in: ")
                print("Training Set: ", round(sum(y_train[:,i])/len(y_train),2) )
                print("Validation Set: ", round(sum(y_val[:,i])/len(y_val),2))
                print("Test Set: ", round(sum(y_test[:,i])/len(y_test),2) , "\n")
        
        
        #######################################################
        #!!!Data augmentation
        #######################################################
        
        
        #Function
        def augment_image(image , rand_bright = True , gaussian = True , denoise = True , flip_rotate = True , contour = False   ):
            
            if rand_bright:
                image = ml.ImageProcessing.random_brightness(image)
            if gaussian:
                image = ml.ImageProcessing.add_gaussian_noise(image , 0.3)
            if denoise:
                image = ml.ImageProcessing.denoise_img(image)
            if flip_rotate:
                image = ml.ImageProcessing.random_rotate_flip(image)
            if contour:
                image = ml.ImageProcessing.contour_mod(image)
                
            return image
        #################################
        
        
        
        lenght = len(x_train)
        blank_class_y = np.zeros((0,len(dictionary)) , dtype = np.uint8)
        if channels == 1:
            blank_class_x = np.zeros((0,img_H,img_W) , dtype = np.uint8)
        else:
            blank_class_x = np.zeros((0,img_H,img_W,channels) , dtype = np.uint8)
            
        if train_dataset_multiplier == 1 :
            pass
            print("No augmentation specified, loading only original images")
        else:
            print("Augmenting dataset:")
            for p in range(train_dataset_multiplier - 1):
            
                x_aug = []
                for i in tqdm(range(lenght)):
                    if np.mean(x_train[i]) == 0:
                        x_train[i,0,0,0] = 1
                    aug_img = augment_image(x_train[i] , parameters[0][1] , parameters[1][1] , parameters[2][1] , parameters[3][1] ,  parameters[4][1])
        
                    x_aug.append(aug_img)
                    
                x_aug = np.asarray(x_aug)
                
                blank_class_x = np.concatenate( (blank_class_x , x_aug) )
                
                blank_class_y = np.concatenate( (blank_class_y , y_train) )
            
            #If contour is True
            if parameters[4][1]:
                x_aug = []
                for i in tqdm(range(lenght)):
                    aug_img = ml.ImageProcessing.contour_mod(x_train[i] , density = 2)
                    x_aug.append(aug_img)
                x_train = np.asarray(x_aug)
               
                
        
        
        
            x_train = np.concatenate((x_train, blank_class_x))
            y_train = np.concatenate((y_train, blank_class_y))
        
    
        
        print("Saving prepared dataset...\n")
        #Saveall of it
        np.save( os.path.join(Data_toRun_directory , "x_train.npy" ) , x_train) 
        np.save( os.path.join(Data_toRun_directory , "y_train.npy" ) , y_train) 
        
        np.save( os.path.join(Data_toRun_directory , "x_val.npy" ) , x_val) 
        np.save( os.path.join(Data_toRun_directory , "y_val.npy" ) , y_val) 
        if not Kaggle_competition_dataset:
            np.save( os.path.join(Data_toRun_directory , "x_test.npy" ) , x_test) 
            np.save( os.path.join(Data_toRun_directory , "y_test.npy" ) , y_test) 
        
        # Write the JSON string to a file
        with open(os.path.join(Data_toRun_directory , "dictionary.json"), 'w') as json_file:
            json_file.write(json.dumps(dictionary))
            
        try:
            del blank_class_x
            del blank_class_y
            del aug_img
            del x_aug
            del lenght
            del p 
            del i
            del json_file
        except:
            print("Could not delete variables, such not exists, skipping...\n")
    
    
    #########################################################################
    #########################################################################
    
    
    #!!! Randomizing data order and adjusting data type for lower memory use 
    #and 0-1 range to increase performance
    #########################################################################
    #########################################################################
    x_train , y_train = skl.utils.shuffle(x_train, y_train, random_state=6)
        
    x_train = x_train.astype(DataType) / 255
    y_train = y_train.astype(DataType)
    
    x_val = x_val.astype(DataType) / 255
    y_val = y_val.astype(DataType)
    if not Kaggle_competition_dataset:
        x_test = x_test.astype(DataType) / 255
        y_test = y_test.astype(DataType)
    #########################################################################
    #########################################################################
    
    
    
    #!!! Defining the architecture of the CNN 
    #and creation of directory based on it and initial parameters
    #########################################################################
    #########################################################################
    
    #Checking if given architecture name is present in library
    model_architecture = f"{model_architecture}"
    
    model_architecture_class = getattr(ml.Architectures.Img_Classification, model_architecture, None)
    
    if model_architecture_class is not None:
        # If the class is found, instantiate the model
        model = model_architecture_class((img_H,img_W,channels) , len(dictionary))
        print("Found architecture named: ",model_architecture,)
    else:
        # If the class is not found, print a message
        model = None
        print("No such model architecture in library")
    
    
    #Creating directory to save weights of trained model and other data about it
    model_name = str(model_architecture + "_bs"+str(batch_size)+".keras")
    
    model_directory =  os.path.join(os.path.dirname(os.getcwd()) , "Models_saved" , str(model_architecture) , form , str(str(img_H)+"x"+str(img_W)) , str("bs"+str(batch_size) + str_parameters)  )
    
    model_weights_directory = os.path.join(model_directory , model_name)
    #Check if directory of trained model is present, if not, create one 
    if not os.path.isdir(model_directory):
        os.makedirs(model_directory)
        print("Creating model directory storage directory...\n")
    #########################################################################
    #########################################################################
    
    
    
    
    #!!! Building and compiling model
    #########################################################################
    #########################################################################
    #Choosing optimizer
    optimizer = tf.keras.optimizers.Adam()
    #Compiling model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    if show_architecture:
        model.summary()
        tf.keras.utils.plot_model(model , os.path.join(os.path.dirname(os.getcwd()) , "Models_saved" , str(model_architecture) , "Model_architecture.png"))

    
    #########################################################################
    #########################################################################
    
    
    
    #!!! Model training
    #########################################################################
    #########################################################################
    model_history_directory = os.path.join(model_directory , "Model_history.csv")
    model , train , starting_epoch = ml.General.Load_model_check_training_progress(model, train, model_weights_directory, model_history_directory)
    
            
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
                                                     min_delta=0.01),
                    #Checkpoint model if performance is increased
                    tf.keras.callbacks.ModelCheckpoint(filepath = model_weights_directory  ,
                                                    monitor = "val_accuracy",
                                                    save_best_only = True,
                                                    verbose = 1),
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
        
    
    #!!! Model results
    #########################################################################
    #########################################################################
    
    #Plot model training history
    Model_history = pd.read_csv(model_history_directory)
    General.Model_training_history_plot_CSV(Model_history)
    
    
    if not Kaggle_competition_dataset:
        #Create confusion matrix
        #Predict classes
        print("Predicting classes based on test set...")
        y_pred = model.predict(x_test)
        
        plt.figure()
        ml.General.Conf_matrix_classification(y_test  ,y_pred , dictionary , normalize = True)
    else:
        #Create confusion matrix
        #Predict classes
        print("Predicting classes based on validation set...")
        y_pred = model.predict(x_val)
        
        plt.figure()
        ml.General.Conf_matrix_classification(y_val ,y_pred , dictionary , normalize = True)
    
    

    if evaluate:
        #Evaluate model
        print("\nModel evaluation train set:")
        model.evaluate(x_train, y_train)
        
        #Evaluate model
        print("\nModel evaluation validation set:")
        model.evaluate(x_val, y_val)
        
        if not Kaggle_competition_dataset:
            #Evaluate model
            print("\nModel evaluation test set:")
            model.evaluate(x_test, y_test)
    
    
    #########################################################################
    #########################################################################
    
    
    
    
if __name__ == "__main__":
   main()   
    
    
    
    
