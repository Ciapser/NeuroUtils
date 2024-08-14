
![alt text](https://github.com/Ciapser/NeuroUtils/blob/main/ReadMe_files/Logo_with_name.svg?raw=true)
# Modular DL project organizer





|           |          |
| :-------- | :------- |
| Package   | [![Static Badge](https://img.shields.io/badge/GitHub_Version-0.2.1-purple)](https://github.com/Ciapser/NeuroUtils) [![Static Badge](https://img.shields.io/badge/PyPi%20Version-0.2.0-blue)](https://pypi.org/project/NeuroUtils/)|
| Info      | [![Static Badge](https://img.shields.io/badge/License-Apache%202.0-green)](https://github.com/Ciapser/NeuroUtils/blob/main/LICENSE.txt)|



## Description
NeuroUtils is the library which goal is to transform your ML/DL project into modular one, and automate process of data preparation and models performance analysis. Library contains also some functions, utilities related with deep learning and image processing.

Its dedicated mostly for scientific and testing purposes and will serve you well if you want to see impact of different data, architectures and parameters configuration on your neural network model.


## Idea

General idea behind the project is to reduce data processing time and to manage different combinations of paramaters automatically so the user don't need to worry about them and can focus on testing different parameters. 
**You just need to change the model as you wish, and NeuroUtils will save, and take care of it.**

- When all steps are finished progress is saved and its possible to experiment with another parameters. All steps are modular and you can inspect data, model and then renew the process.
![alt text](https://github.com/Ciapser/NeuroUtils/blob/main/ReadMe_files/FlowChart.png?raw=true)

- Library is checking if in current run, any identical steps have been performed in the past. If so, they are skipped, to save time and moves to the next step, as shown at the **Workflow chart** below.
![alt text](https://github.com/Ciapser/NeuroUtils/blob/main/ReadMe_files/Workflow.jpg?raw=true)


## Use
To start using library you need to create the script in chosen folder, and run it. Necessary folders will be created automatically. 
##### Below is simple instruction through the steps

#### Importing library:
```python
from NeuroUtils import Core
from NeuroUtils import Architectures
#Eventually if you want to go deeper, there is library section  with  more basic functions:
from NeuroUtils import ML_assets as ml
```
#### Data preparation:
Creating class of the project and setting data directory\
```python
Example_Project = Core.Project.Classification_Project(Database_Directory = "Your\DataBase\Folder")
```
    
    
    

## Feedback

If you have any feedback about the project, you can reach me on my github profile


### Disclaimer
Project is made mostly for educational purposes, to learn and deploy new things into real project, so especially at its current state it may be not best optimized and other official solutions can be more suited to some problems. However its great fun for me and I will develop and try to make it as good as I can.