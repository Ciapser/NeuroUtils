
![alt text](https://github.com/Ciapser/NeuroUtils/blob/main/ReadMe_files/Logo_with_name.svg?raw=true)
# Modular Deep Learning project organizer





|           |          |
| :-------- | :------- |
| Package   | [![Static Badge](https://img.shields.io/badge/GitHub_Version-0.2.1-purple)](https://github.com/Ciapser/NeuroUtils) [![Static Badge](https://img.shields.io/badge/PyPi%20Version-0.2.0-blue)](https://pypi.org/project/NeuroUtils/)|
| Info      | [![Static Badge](https://img.shields.io/badge/License-Apache%202.0-green)](https://github.com/Ciapser/NeuroUtils/blob/main/LICENSE.txt)|



## Description
NeuroUtils is the library which goal is to transform your ML/DL project into modular one, and automate process of data preparation and models performance analysis. Library contains also some functions, utilities related with deep learning and image processing.

Its dedicated mostly for scientific and testing purposes and will serve you well if you want to see impact of different data, architectures and parameters configuration on your neural network model.

Currently it officially supports classification problems and other ones may not be best optimized to work with. But it's planned to extend support for object detection and segmentation in the future.


## Idea

General idea behind the project is to reduce data processing time and to manage different combinations of paramaters automatically so the user don't need to worry about them and can focus on testing different parameters. 
**You just need to change the model as you wish, and NeuroUtils will save, and take care of it.**

- When all steps are finished progress is saved and its possible to experiment with another parameters. All steps are modular and you can inspect data, model and then renew the process.
![alt text](https://github.com/Ciapser/NeuroUtils/blob/main/ReadMe_files/FlowChart_reduced_size.png?raw=true)

- Library is checking if in current run, any identical steps have been performed in the past. If so, they are skipped, to save time and moves to the next step, as shown at the **Workflow chart** below.
![alt text](https://github.com/Ciapser/NeuroUtils/blob/main/ReadMe_files/Workflow.jpg?raw=true)


## Instalation
## Use
To start using library you need to create the script in chosen folder, and run it. Necessary folders will be created automatically. 
##### Below is simple instruction through the steps

#### **Importing library**:
```python
from NeuroUtils import Core
from NeuroUtils import Architectures
#Eventually if you want to go deeper, there is library section  with  more basic functions:
from NeuroUtils import ML_assets as ml
```

#### **Project Initialization**:
Creating class of the project and setting data directory
```python
Example_Project = Core.Project.Classification_Project(Database_Directory = "Your\DataBase\Folder")
```

#### **Data preparation**:
Initializating data from main database folder to project folder 
```python
Example_Project.Initialize_data(Img_Height = 32, Img_Width = 32, Grayscale = False)
```

#### **Data processing**:
Loading prepared data and processing it
```python
#Loading and merging data to trainable dataset, with optional reduction of the size class
x, y, dictionary = Example_Project.Load_and_merge_data(Reduced_class_size= None)

#Processing data by splitting it to train,val and test set and augmenting
x_train, y_train, x_val, y_val,x_test,y_test = Example_Project.Process_data(X = x,
                                                                            Y = y,
                                                                            Val_split = 0.1,
                                                                            Test_split = 0.1,
                                                                            DataSet_multiplier = 1,
                                                                            DataType = "float32",
                                                                            FlipRotate = False,
                                                                            RandBright = False,
                                                                            Gaussian_noise = False,
                                                                            Denoise = False,
                                                                            Contour = False
                                                                            )
```
    
#### **Data injection**:
Saving data into class
```python
Example_Project.Save_Data(x_train = x_train,
                          y_train = y_train,
                          x_val = x_val,
                          y_val = y_val,
                          dictionary = dictionary,
                          x_test = x_test,
                          y_test = y_test
                          )
```   

#### **Model Preparation**:
There are several architectures in the library, however **its reccomended to use own architecture**, or import it from another library (tensorflow,keras etc.), as these present in the "Architectures" module are not well tested and are used mainly for testing.
```python
h = Example_Project.IMG_H
w = Example_Project.IMG_W
c = Example_Project.CHANNELS
n_classes = Example_Project.N_CLASSES

#Loading custom architecture
model = Architectures.Img_Classification.AnimalNet_v32(shape = (h,w,c), n_classes =  n_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```    

#### **Training model**:
#Training of the model. It can load previously saved model from project folder or train from scratch.
```python
model = Example_Project.Initialize_weights_and_training(Model = model,
                                                        Architecture_name = 'AnimalNet_v32',
                                                        Epochs= 300,
                                                        Batch_size = 8,
                                                        Train = True,
                                                        Patience = 30,
                                                        Min_delta_to_save = 0.001,
                                                        Device = "GPU",
                                                        Checkpoint_monitor = "val_loss",
                                                        Checkpoint_mode = "min",
                                                        add_config_info = None
                                                        )
```  
    
#### **Model Evaluation**:
Showing and saving basic results of the current training session
```python
Example_Project.Initialize_results(show_plots = False, save_plots = True, Evaluate = False)
```    

#### **Model Analysis**:
Performing high detail analysis of all models trained
```python
Core.Utils.Models_analysis(show_plots = False, save_plots = True)
```  
## Results
- **F scores analysis plots** [![Static Badge](https://img.shields.io/badge/Full_HD-F_scores_analysis-green)](https://github.com/Ciapser/NeuroUtils/blob/main/ReadMe_files/F_scores.png)

![alt text](https://github.com/Ciapser/NeuroUtils/blob/main/ReadMe_files/F_scores_reduced_size.png?raw=true)

- **Train_history** [![Static Badge](https://img.shields.io/badge/Full_HD-Train_History-green)](https://github.com/Ciapser/NeuroUtils/blob/main/ReadMe_files/Train_history.png)

![alt text](https://github.com/Ciapser/NeuroUtils/blob/main/ReadMe_files/Train_history_reduced_size.png?raw=true)

- **Confusion matrix** [![Static Badge](https://img.shields.io/badge/Full_HD-Conf_Matrix-green)](https://github.com/Ciapser/NeuroUtils/blob/main/ReadMe_files/Confusion_matrix.png)

![alt text](https://github.com/Ciapser/NeuroUtils/blob/main/ReadMe_files/Confusion_matrix_reduced_size.png?raw=true)

- **Model PDF report** [![Static Badge](https://img.shields.io/badge/Full_HD-PDF_Report-green)](https://github.com/Ciapser/NeuroUtils/blob/main/ReadMe_files/Model_preview.png)

![alt text](https://github.com/Ciapser/NeuroUtils/blob/main/ReadMe_files/Model_preview_reduced_size.png?raw=true)

## **And more not shown on the images**:
- Analysis over train,
- Another metrics analysis
- Model architecture preview
- Model overfit analysis
- Stored metadata about models for your own analysis
- More in the future




## Feedback

If you have any feedback about the project, you can reach me on my github profile


### Disclaimer
Project is made mostly for educational purposes, to learn and deploy new things into real project, so especially at its current state it may be not best optimized and other official solutions can be more suited to some problems. However its great fun for me and I will develop and try to make it as good as I can.