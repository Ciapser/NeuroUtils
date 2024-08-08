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
import gc
import ast
from scipy.signal import savgol_filter

import matplotlib as mpl
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib import colors
import re
from PIL import Image, ImageDraw
from io import BytesIO
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
import io
from reportlab.lib.utils import ImageReader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


class Utils:
    def Analysis_plot(metrics_train, metrics_val, x_labels, plot_title, save_plots, show_plots, analysis_folder_path = "Analysis", clean_title = False,maximum_y = None,round_data = True):
        #current_script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        #if analysis_folder_path == "Analysis":
            #analysis_folder_path = os.path.join(current_script_dir,"Analysis")
            
        
        rows = 0
        if metrics_train is not None:
            create_train = True
            rows+=1
        else:
            create_train = False
        if metrics_val is not None:
            create_val = True
            rows+=1
        else:
            create_val = False
        if not(metrics_train or metrics_val):
            print("No data provided, skipping analysis plot...")
            return
        
        if rows >1:
            fig, axes = plt.subplots(nrows=rows, ncols=1, figsize=(2*len(x_labels), 10))  # Two subplots, one above the other
        
        else:
            fig, axes = plt.subplots(figsize=(2*len(x_labels), 10))  # Two subplots, one above the other
        
        if create_train:
            # Plot for training data
            if create_val:
                ax = axes[0]
            else:
                ax = axes
            x = np.arange(len(x_labels))  # the label locations
            width = 0.13  # the width of the bars, reduced to make clusters tighter
            multiplier = 0
            min_y = 1
            max_y = 0
            for attribute, value in metrics_train.items():
                offset = width * multiplier
                if round_data:
                    value = [round(v, 3) for v in value]
                rects = ax.bar(x + offset, value, width, label=attribute, edgecolor='black')
                labels = ax.bar_label(rects, padding=3)
                for label in labels:
                    label.set_fontsize('small')  # Set the fontsize here
                multiplier += 1
                if min_y > min(value):
                    min_y = min(value)
                if max_y < max(value):
                    max_y = max(value)
            
            ax.set_ylabel('Value\n Range: [0-1]')
            if clean_title:
                ax.set_title(plot_title)
            else:
                ax.set_title(plot_title + ' (Train)')
            ax.set_xticks(x + width * (multiplier - 1) / 2)
            ax.set_xticklabels(x_labels, fontsize=7)
            ax.legend(loc='upper left', ncols=1, fontsize='small', bbox_to_anchor=(1, 1))  # Position the legend outside the plot
            min_y = (min_y * 0.9)
            max_y = (max_y * 1.1)
            if maximum_y is not None:
                if maximum_y<max_y:
                    max_y = maximum_y
            ax.set_ylim(min_y, max_y)
            ax.set_axisbelow(True)
            ax.grid(color='gray', linestyle='dashed')
            
        if create_val:
            # Plot for validation data
            if create_train:
                ax = axes[1]
            else:
                ax = axes
            x = np.arange(len(x_labels))  # the label locations
            width = 0.13  # the width of the bars, reduced to make clusters tighter
            multiplier = 0
            min_y = 1
            max_y = 0
            for attribute, value in metrics_val.items():
                offset = width * multiplier
                if round_data:
                    value = [round(v, 3) for v in value]
                rects = ax.bar(x + offset, value, width, label=attribute, edgecolor='black')
                labels = ax.bar_label(rects, padding=3)
                for label in labels:
                    label.set_fontsize('small')  # Set the fontsize here
                multiplier += 1
                if min_y > min(value):
                    min_y = min(value)
                if max_y < max(value):
                    max_y = max(value)

            ax.set_ylabel('Value\n Range: [0-1]')
            if clean_title:
                ax.set_title(plot_title)
            else:
                ax.set_title(plot_title + ' (Validation)')
            ax.set_xticks(x + width * (multiplier - 1) / 2)
            ax.set_xticklabels(x_labels, fontsize=7)
            ax.legend(loc='upper left', ncols=1, fontsize='small', bbox_to_anchor=(1, 1))  # Position the legend outside the plot
            min_y = (min_y * 0.9)
            max_y = (max_y * 1.1)
            if maximum_y is not None:
                if maximum_y<max_y:
                    max_y = maximum_y
            ax.set_ylim(min_y, max_y)
            ax.set_axisbelow(True)
            ax.grid(color='gray', linestyle='dashed')
            
            # Adjust subplot parameters to reduce whitespace and ensure the legend is fully visible
            plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1, hspace=0.5)  # Adjust right to leave space for the legend, hspace for vertical space between plots

        # Show the plot
        if show_plots:
            plt.show()
        if save_plots:
            file_name = plot_title + ".png"
            performance_plots_path = os.path.join(analysis_folder_path,"Performance_plots")
            if not os.path.isdir(performance_plots_path):
                os.makedirs(performance_plots_path)
            basic_metrics_path = os.path.join(performance_plots_path, file_name)
            fig.savefig(basic_metrics_path, bbox_inches='tight', dpi=300)

        if not show_plots:
            plt.close()
    
    
        ##########################################################################################    
    def Analysis_over_train(models_metric, val_models_metric, x_labels, plot_title, 
                                             show_plots, save_plots, analysis_folder_path = "Analysis", 
                                             window_length=20, polyorder=2,min_y = 0,max_y = 1):
        """
        Plot smoothed training and validation accuracy for multiple models and highlight the maximum value.
    
        Parameters:
        - models_metric: list of np.ndarray, List of training metric arrays from different models.
        - val_models_metric: list of np.ndarray, List of validation metric arrays from different models.
        - x_labels: list of str, Labels for each model to use in the legend.
        - plot_title: str, Title of the plot.
        - show_plots: bool, Whether to display the plots.
        - save_plots: bool, Whether to save the plots.
        - train_window_length: int, Window length for the Savitzky-Golay filter for training data (must be odd).
        - train_polyorder: int, Polynomial order for the Savitzky-Golay filter for training data.
        - val_window_length: int, Window length for the Savitzky-Golay filter for validation data (must be odd).
        - val_polyorder: int, Polynomial order for the Savitzky-Golay filter for validation data.
        """
        
        num_models = len(models_metric)
        colors = plt.cm.viridis(np.linspace(0, 1, num_models))  # Generate distinct colors for each model
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), sharex=True)
        
        for i, (train_acc, val_acc) in enumerate(zip(models_metric, val_models_metric)):
            epochs = np.arange(0, len(train_acc))
            # Adjust window length if it is larger than the number of data points for training data
            if window_length >= len(train_acc):
                window_length = len(train_acc) if len(train_acc) % 2 != 0 else len(train_acc) - 1
            
            # Adjust window length if it is larger than the number of data points for validation data
            if window_length >= len(val_acc):
                window_length = len(val_acc) if len(val_acc) % 2 != 0 else len(val_acc) - 1

            # Smoothing the training data using Savitzky-Golay filter
            train_acc_smoothed = savgol_filter(train_acc, window_length=window_length, polyorder=polyorder)

            # Smoothing the validation data using Savitzky-Golay filter
            val_acc_smoothed = savgol_filter(val_acc, window_length=window_length, polyorder=polyorder)

            # Find the maximum value and its epoch for training data
            max_train_val = np.max(train_acc)
            max_train_epoch = np.argmax(train_acc) 
            smoothed_train_val_at_max = train_acc_smoothed[max_train_epoch ]
            
            # Find the maximum value and its epoch for validation data
            max_val_val = np.max(val_acc)
            max_val_epoch = np.argmax(val_acc) 
            smoothed_val_val_at_max = val_acc_smoothed[max_val_epoch]
            
            # Plot smoothed training data
            ax1.plot(epochs, train_acc_smoothed, color=colors[i])
            
            # Highlight the maximum value for training data
            ax1.scatter(max_train_epoch, max_train_val, color=colors[i], zorder=5)
            ax1.plot([max_train_epoch, max_train_epoch], [smoothed_train_val_at_max, max_train_val], color=colors[i], linestyle='-')
            
            # Plot smoothed validation data
            ax2.plot(epochs, val_acc_smoothed, color=colors[i])
            
            # Highlight the maximum value for validation data
            ax2.scatter(max_val_epoch, max_val_val, color=colors[i], zorder=5)
            ax2.plot([max_val_epoch, max_val_epoch], [smoothed_val_val_at_max, max_val_val], color=colors[i], linestyle='-')
    
            # Add the marker to the legend with label
            ax1.scatter([], [], color=colors[i], label=x_labels[i], marker='s')
            ax2.scatter([], [], color=colors[i], label=x_labels[i], marker='s')
        
        
        
        # Labels and legend for training plot
        ax1.set_title(plot_title + ' - Training')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.grid(True)
        ax1.set_ylim(min_y, max_y)
        ax1.set_xlabel('Epochs')
        
        # Labels and legend for validation plot
        ax2.set_title(plot_title + ' - Validation')
        ax2.grid(True)
        ax2.set_ylim(min_y, max_y)
        ax2.set_xlabel('Epochs')
        

        # Adjust subplot parameters to reduce whitespace and ensure the legend is fully visible
        plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1, hspace=0.5)  # Adjust right to leave space for the legend, hspace for vertical space between plots
    
        # Show the plot
        if show_plots:
            plt.show()
        if save_plots:
            file_name = plot_title + ".png"

            history_plots_path = os.path.join(analysis_folder_path,"Train_history_plots")
            if not os.path.isdir(history_plots_path):
                os.makedirs(history_plots_path)
            basic_metrics_path = os.path.join(history_plots_path, file_name)
            fig.savefig(basic_metrics_path, bbox_inches='tight', dpi=300)
    
        if not show_plots:
            plt.close()    
    

    def Models_analysis(models_directory = "Models_saved",analysis_folder_directory = "Analysis",processed_data_folder = "DataSet_Processed" , show_plots = False, save_plots = True):
        current_script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        if models_directory == "Models_saved":
            models_directory = os.path.join(current_script_dir,"Models_saved")
        if analysis_folder_directory == "Analysis":
            analysis_folder_directory = os.path.join(current_script_dir,"Analysis")
        if processed_data_folder == "DataSet_Processed":
            processed_data_folder = os.path.join(current_script_dir,"DataSet_Processed")
        
        print("----------------------------------------------------------------------------")
        print("Starting trained models analysis...")
        
        #Checking for folder names, filtering if something in the folder is not another folder
        unfiltered_folders = []
        for item in os.listdir(os.path.join(models_directory)):
            item_path = os.path.join(models_directory, item)
            if os.path.isdir(item_path):
                unfiltered_folders.append(item)
                
        #checks if model folder have necessary files, if not. Put it out of the list
        folders = []
        for item in unfiltered_folders:
            contains_all_files = True
            files = ["Model_best.keras", "Model.keras", "Model.keras_score.json", "Model_history.csv", "Model_parameters.json"]
            for file in files:
                if not os.path.isfile(os.path.join(models_directory,item,file)):
                    print("Model: ",item,"do not contain file: ",file,"and wont be considered in the analysis")
                    contains_all_files = False

            if contains_all_files:
                folders.append(item)
                
        Data_path = os.path.join(analysis_folder_directory,"Data.csv")
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

                 "N_classes",
                 "Img_number",
                 "Aug_mark",
                 "Img_Dtype",
                 "Form",
                 "Img_H",
                 "Img_W",
                  
                 "Loss_train",
                 "Loss_val",
                 
                 "Train_y_true",
                 "Train_y_pred",
                 "Val_y_true",
                 "Val_y_pred",
                 
                 "Additional_info"
                 ]
            Data = pd.DataFrame(columns = c)
        else:
            Data = pd.read_csv(Data_path)

        #Determine if this model is already present in the data
        for model in folders:
            print("----------------------------------------------------------------------------")
            print("Model: ",model)
            Model_history = pd.read_csv(os.path.join(models_directory,model,"Model_history.csv"))
            if Data["Model_ID"].isin([model]).any():
                print("Model present in Data")
                #To update if epochs has been changed
                model_epoch = Model_history["epoch"].iloc[-1]
                idx = Data['Model_ID'].eq(model).argmax()
                analysis_epoch = Data["Epochs_trained"][idx]

                if model_epoch>analysis_epoch:
                    fill_data = True
                    print("Model has been trained during last analysis, updating...")
                else:
                    fill_data = False
                    print("Data is up to date, skipping...")
  
                
            else:
                fill_data = True
                print("Model has not been processed yet, starting analysis")
                
                
            if fill_data:
                
                param_path = os.path.join(models_directory,model,"Model_parameters.json")
                with open(param_path, 'r') as json_file:
                    param_data = json.load(json_file)

                Model_WorkName = param_data["User Architecture Name"]
                Model_Parameters = param_data["Total Parameters"]
                Model_Dtype = param_data["Model Datatype"]
                Batch_Size = param_data["Batch Size"]
                Checkpoint_mode = param_data["Checkpoint mode"]
                Checkpoint_monitor = param_data["Checkpoint monitor"]
                Epochs_Trained = Model_history["epoch"].iloc[-1]
                
                
                if Checkpoint_mode == "min":
                    best_metric = Model_history[Checkpoint_monitor].idxmin()
                elif Checkpoint_mode =="max":
                    best_metric = Model_history[Checkpoint_monitor].idxmax()
                else:
                    print("Incorrect checkpoint_mode!")
                    

                Epochs_toBest = (Model_history[Checkpoint_monitor] == Model_history[Checkpoint_monitor][best_metric]).idxmax()

                Optimizer_data = param_data["Optimizer Parameters"]
                Loss_type = param_data["Loss function"]

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
                
                img_mark = str(Img_H)+"x"+str(Img_W)+"_"+Form
                aug_mark = param_data["Augmentation Mark"]
                split_mark = "Val_"+str(param_data["Validation split"]) + "  Test_"+str(param_data["Test split"])
                processed_data_dir = os.path.join(processed_data_folder,img_mark,aug_mark,split_mark)
                
                
                Keras_model = tf.keras.models.load_model(os.path.join(models_directory,model,"Model_best.keras"))
                try:
                    x_train = (np.load(os.path.join(processed_data_dir,"x_train.npy"))/255).astype(Img_Dtype)
                    y_train = np.load(os.path.join(processed_data_dir,"y_train.npy"))
                    print("Calculating train loss...")
                    Loss_train,_ = Keras_model.evaluate(x_train,y_train)
                    
                    train_y_true = np.argmax(y_train,axis = 1).astype(int).tolist()
                    
                    train_y_pred = Keras_model.predict(x_train)
                    train_y_pred = np.argmax(train_y_pred,axis = 1).astype(int).tolist()

                except:
                    print("No train set found")
                    Loss_train = None


                try:
                    x_val = (np.load(os.path.join(processed_data_dir,"x_val.npy"))/255).astype(Img_Dtype)
                    y_val = np.load(os.path.join(processed_data_dir,"y_val.npy"))
                    print("Calculating validation loss...")
                    Loss_val,_ = Keras_model.evaluate(x_val,y_val)
                    
                    val_y_true = np.argmax(y_val,axis = 1).astype(int).tolist()
                    
                    val_y_pred = Keras_model.predict(x_val)
                    val_y_pred = np.argmax(val_y_pred,axis = 1).astype(int).tolist()

                except:
                    print("No validation set found")
                    Loss_val = None

                    

                Additional_info = param_data["Additional information"]
                


                if os.path.isfile(Data_path):       
                    try:
                        #Such model is present and its saved in the same row
                        idx = int(Data.index[Data['Model_ID']==model].tolist()[0])
                        
                    except:
                        #Such model is not present yet and is saved in the new row
                        idx = len(Data['Model_ID'])
                        
                else:
                    print("Data analysis file is not present yet, crating one...")
                    idx = 0
                row_list = [   model,
                                    Model_WorkName,
                                    Model_Parameters,
                                    Model_Dtype,
                                    Batch_Size,
                                    Epochs_Trained,
                                    Epochs_toBest,
                                 
                                    Optimizer_data,
                                    Loss_type,
        
                                    N_classes,
                                    Img_number,
                                    Aug_mark,
                                    Img_Dtype,
                                    Form,
                                    Img_H,
                                    Img_W,
                                  
                                    Loss_train,
                                    Loss_val,
                                    
                                    train_y_true,
                                    train_y_pred,
                                    val_y_true,
                                    val_y_pred,

                                    Additional_info
                                  ]
                row_list = np.array(row_list,dtype = "object")
                Data.loc[idx] = row_list
                
                
                
                del x_train,y_train,x_val,y_val
                del Model_WorkName, Model_Parameters, Model_Dtype, Batch_Size, Epochs_Trained, Epochs_toBest,Optimizer_data,Loss_type
                del N_classes,Img_number,Aug_mark,Img_Dtype,Form,Img_H,Img_W,Loss_train,Loss_val,Additional_info
                                    
            
                tf.keras.backend.clear_session()
                gc.collect()
                #At the end of Data creation
                Data.to_csv(Data_path,index = False)    
        
        ####################################################################################################
        Data = pd.read_csv(Data_path, converters={'Train_y_true': pd.eval,
                                                  'Train_y_pred': pd.eval,
                                                  'Val_y_true': pd.eval,
                                                  'Val_y_pred': pd.eval
                                                  })


        #Calculate accuracy
        train_models_high_level_metrics = {}
        val_models_high_level_metrics = {}
        for i in range(len(Data)):
            n_classes = Data["N_classes"][i]
            #Train data
            train_y_true = Data["Train_y_true"][i]
            train_y_pred = Data["Train_y_pred"][i]
            
            train_y_true = ml.General.OneHot_decode(train_y_true,n_classes)
            train_y_pred = ml.General.OneHot_decode(train_y_pred,n_classes)

            
            train_metrics = ml.General.calculate_main_metrics(y_true_one_hot = train_y_true,
                                                              y_pred_prob_one_hot = train_y_pred
                                                              )
            #Calculating ROC metrics
            train_metrics["ROC_scores"] = ml.General.compute_multiclass_roc_auc(y_true = train_y_true,
                                                                                y_pred = train_y_pred
                                                                                )
            
            train_models_high_level_metrics[Data["Model_ID"][i]] = train_metrics

            
            
            #Val data
            val_y_true = Data["Val_y_true"][i]
            val_y_pred = Data["Val_y_pred"][i]
            
            val_y_true = ml.General.OneHot_decode(val_y_true,n_classes)
            val_y_pred = ml.General.OneHot_decode(val_y_pred,n_classes)
            
            val_metrics = ml.General.calculate_main_metrics(y_true_one_hot = val_y_true,
                                                              y_pred_prob_one_hot = val_y_pred
                                                              )
            
            val_metrics["ROC_scores"] = ml.General.compute_multiclass_roc_auc(y_true = val_y_true,
                                                                                y_pred = val_y_pred
                                                                                )
            val_models_high_level_metrics[Data["Model_ID"][i]] = val_metrics
            


        #Train lists
        t_acc = []
        t_prec = []
        t_rec = []
        t_spec = []
        t_f05 = []
        t_f1 = []
        t_f2 =[]
        t_bal_acc = []
        t_ROC_AUC = []
        #Val lists
        v_acc = []
        v_prec = []
        v_rec = []
        v_spec = []
        v_f05 = []
        v_f1 = []
        v_f2 =[]
        v_bal_acc = []
        v_ROC_AUC = []
        
        harmonic_ov = []
        
        x_labels = []
        

        
        for model in folders:
            #Train data
            t_acc.append(train_models_high_level_metrics[model]['accuracy'])
            t_prec.append(train_models_high_level_metrics[model]['precision'])
            t_rec.append(train_models_high_level_metrics[model]['recall'])
            t_spec.append(train_models_high_level_metrics[model]['specificity'])
            
            t_f05.append(train_models_high_level_metrics[model]['f0_5_score'])
            t_f1.append(train_models_high_level_metrics[model]['f1_score'])
            t_f2.append(train_models_high_level_metrics[model]['f2_score'])
            
            t_bal_acc.append(train_models_high_level_metrics[model]['balanced_accuracy'])
            t_ROC_AUC.append(train_models_high_level_metrics[model]['ROC_scores'][0])
            
            #Val data
            v_acc.append(val_models_high_level_metrics[model]['accuracy'])
            v_prec.append(val_models_high_level_metrics[model]['precision'])
            v_rec.append(val_models_high_level_metrics[model]['recall'])
            v_spec.append(val_models_high_level_metrics[model]['specificity'])
            
            v_f05.append(val_models_high_level_metrics[model]['f0_5_score'])
            v_f1.append(val_models_high_level_metrics[model]['f1_score'])
            v_f2.append(val_models_high_level_metrics[model]['f2_score'])
            
            v_bal_acc.append(val_models_high_level_metrics[model]['balanced_accuracy'])
            v_ROC_AUC.append(val_models_high_level_metrics[model]['ROC_scores'][0])
            
            #Overtraining_data
            acc_ov = (ml.General.compute_overtrain_metric(train_models_high_level_metrics[model]['accuracy'], val_models_high_level_metrics[model]['accuracy']))
            bal_acc_ov = (ml.General.compute_overtrain_metric(train_models_high_level_metrics[model]['balanced_accuracy'], val_models_high_level_metrics[model]['balanced_accuracy']))
            f1_ov = (ml.General.compute_overtrain_metric(train_models_high_level_metrics[model]['f1_score'], val_models_high_level_metrics[model]['f1_score']))
           
            harmonic_ov_mean = np.array([acc_ov,bal_acc_ov,f1_ov])
            
            harmonic_ov_mean = len(harmonic_ov_mean) / np.sum(1/harmonic_ov_mean)
            
            
            harmonic_ov.append(harmonic_ov_mean)



            #Model plot display name
            idx = int(Data.index[Data['Model_ID']==model].tolist()[0])
            
            Name_label = Data["Model_WorkName"][idx]
            
            Model_label = Data["Model_Parameters"][idx]
            Model_Dtype = Data["Model_Dtype"][idx]
            if len(str(Model_label)) >3 and len(str(Model_label))<6:
                Model_label = str(round(Model_label/1e3,1))+ "K_"+Model_Dtype
            elif len(str(Model_label)) >=6:
                Model_label = str(round(Model_label/1e6,1))+ "M_"+Model_Dtype
                
            
            Img_H = Data["Img_H"][idx]
            Img_W = Data["Img_W"][idx]
            Img_Dtype = Data["Img_Dtype"][idx]
            Img_Form = Data["Form"][idx]
            
            Img_label = str(Img_H) +"x"+ str(Img_W)+ " "+Img_Form+"_"+Img_Dtype
            
            bs = str(Data["Batch_size"][idx])
            Opt = ast.literal_eval(Data["Optimizer_data"][idx])['name']
            loss = str(Data["Loss_type"][idx])
            Train_label = "Opt:"+Opt+" B_size: "+bs +"\nLoss: "+loss
            
            Epochs_trained = str(Data["Epochs_trained"][idx])
            Epochs_best = str(Data["Epochs_toBest"][idx])
            
            Epochs_label = 'Epochs best/trained: '+Epochs_best+"/"+Epochs_trained
            
            ID_label = "Short ID: "+model[0:5]
            
            Plot_label = Name_label+" "+Model_label+"\n"+Img_label+"\n"+Train_label+"\n"+Epochs_label+"\n"+ID_label

            x_labels.append(Plot_label)
            
        #Plot dta for train
        t_models_basic_metrics = {
            'Accuracy': t_acc,
            'Precision': t_prec,
            'Recall': t_rec,
            'Specificity': t_spec
        }
        
        t_models_f_scores = {
            'F05_score': t_f05,
            'F1_score': t_f1,
            'F2_score': t_f2,
        }
        
        t_Balanced_metrics = {
            'Balanced_accuracy': t_bal_acc,
            'ROC_AUC': t_ROC_AUC,
        }
        
        #Plot dta for val v_
        v_models_basic_metrics = {
            'Accuracy': v_acc,
            'Precision': v_prec,
            'Recall': v_rec,
            'Specificity': v_spec
        }
        
        v_models_f_scores = {
            'F05_score': v_f05,
            'F1_score': v_f1,
            'F2_score': v_f2,
        }
        
        v_Balanced_metrics = {
            'Balanced_accuracy': v_bal_acc,
            'ROC_AUC': v_ROC_AUC,
        }
        
        Another_metrics = {
            'Harmonic_overtrain': harmonic_ov

        }

        Utils.Analysis_plot(metrics_train = t_models_basic_metrics,
                           metrics_val = v_models_basic_metrics,
                           x_labels = x_labels,
                           plot_title = "Basic metrics",
                           save_plots = save_plots,
                           show_plots =show_plots,
                           maximum_y = 1)

        
        Utils.Analysis_plot(metrics_train = t_models_f_scores,
                           metrics_val = v_models_f_scores,
                           x_labels = x_labels,
                           plot_title = "F scores",
                           save_plots = save_plots,
                           show_plots =show_plots,
                           maximum_y = 1)
        
        Utils.Analysis_plot(metrics_train = t_Balanced_metrics,
                           metrics_val = v_Balanced_metrics,
                           x_labels = x_labels,
                           plot_title = "Inbalance resistant metrics",
                           save_plots = save_plots,
                           show_plots =show_plots,
                           maximum_y = 1)
        
        Utils.Analysis_plot(metrics_train = Another_metrics,
                           metrics_val = None,
                           x_labels = x_labels,
                           plot_title = "Other model metrics",
                           save_plots = save_plots,
                           show_plots =show_plots,
                           clean_title = True,
                           round_data = False)
        
#!!!
        ####################################
        #Analysis metrics over training
        print("\n\n")
        print("Training history analysis...")

        Train_history_path = os.path.join(analysis_folder_directory,"Train_history_Data.feather")
        #Create dataframe with analysys if it does not exists yet
        if not os.path.isfile(Train_history_path):
            c = ['Model_ID',
                 'train_accuracy',
                 'train_precision',
                 'train_recall',
                 'train_f1_score',
                 'train_f2_score',
                 'train_f0_5_score',
                 'train_specificity',
                 'train_balanced_accuracy',
                 'train_roc_auc',
                 
                 'val_accuracy',
                 'val_precision',
                 'val_recall',
                 'val_f1_score',
                 'val_f2_score',
                 'val_f0_5_score',
                 'val_specificity',
                 'val_balanced_accuracy',
                 'val_roc_auc',
                 
                 ]
            Train_history_Data = pd.DataFrame(columns = c)
        else:
            Train_history_Data = pd.read_feather(Train_history_path)

        #return Train_history_Data
        #Determine if this model is already present in the data
        for model in folders:
            print("----------------------------------------------------------------------------")
            print("Model: ",model)
            Model_history = pd.read_csv(os.path.join(models_directory,model,"Model_history.csv"))
            if Train_history_Data["Model_ID"].isin([model]).any():
                print("Model present in Data")
                #To update if epochs has been changed
                model_epoch = Model_history["epoch"].iloc[-1]
                idx = Train_history_Data['Model_ID'].eq(model).argmax()
                analysis_epoch = len(Train_history_Data["train_accuracy"][idx])-1
                
                if model_epoch>analysis_epoch:
                    fill_data = True
                    print("Model has been trained during last analysis, updating...")
                else:
                    fill_data = False
                    print("Data is up to date, skipping...")
  
            else:
                fill_data = True
                print("Model has not been processed yet, starting analysis")
        
        


            if fill_data:
                param_path = os.path.join(models_directory,model,"Model_parameters.json")
                with open(param_path, 'r') as json_file:
                    param_data = json.load(json_file)
                    
                N_classes = param_data["Number of Classes"]

                metric_path = os.path.join(models_directory,model ,"Model_metrics.csv")
                model_metrics = pd.read_csv(metric_path)

                
                
                for col in model_metrics.columns:
                    if col != "epoch":
                        model_metrics[col] = model_metrics[col].apply(ast.literal_eval)   


                
                train_accuracy = []
                train_precision = []
                train_recall = []
                train_f1_score = []
                train_f0_5_score = []
                train_f2_score = []
                train_specificity = []
                train_balanced_accuracy = []
                train_roc_auc = []
                
                val_accuracy = []
                val_precision = []
                val_recall = []
                val_f1_score = []
                val_f0_5_score = []
                val_f2_score = []
                val_specificity = []
                val_balanced_accuracy = []
                val_roc_auc = []
                #Train 
                for k in tqdm(range(len(model_metrics))):
                    
                    train_true = model_metrics['train_true'][k]
                    train_pred = model_metrics['train_pred'][k]
                    
                    train_true = ml.General.OneHot_decode(train_true, N_classes)
                    train_pred = ml.General.OneHot_decode(train_pred, N_classes)
                    
                    train_calc_metrics = ml.General.calculate_main_metrics(train_true, train_pred)

                    train_accuracy.append(train_calc_metrics["accuracy"])
                    train_precision.append(train_calc_metrics["precision"])
                    train_recall.append(train_calc_metrics["recall"])
                    train_f1_score.append(train_calc_metrics["f1_score"])
                    train_f0_5_score.append(train_calc_metrics["f0_5_score"])
                    train_f2_score.append(train_calc_metrics["f2_score"])
                    train_specificity.append(train_calc_metrics["specificity"])
                    train_balanced_accuracy.append(train_calc_metrics["balanced_accuracy"])
                    train_roc_auc.append(ml.General.compute_multiclass_roc_auc(train_true, train_pred)[0])
                    

                    
                #Validation
                for k in tqdm(range(len(model_metrics))):
                    val_true = model_metrics['val_true'][k]
                    val_pred = model_metrics['val_pred'][k]
                    
                    val_true = ml.General.OneHot_decode(val_true, N_classes)
                    val_pred = ml.General.OneHot_decode(val_pred, N_classes)
                    
                    val_calc_metrics = ml.General.calculate_main_metrics(val_true, val_pred)

                    val_accuracy.append(val_calc_metrics["accuracy"])
                    val_precision.append(val_calc_metrics["precision"])
                    val_recall.append(val_calc_metrics["recall"])
                    val_f1_score.append(val_calc_metrics["f1_score"])
                    val_f0_5_score.append(val_calc_metrics["f0_5_score"])
                    val_f2_score.append(val_calc_metrics["f2_score"])
                    val_specificity.append(val_calc_metrics["specificity"])
                    val_balanced_accuracy.append(val_calc_metrics["balanced_accuracy"])
                    val_roc_auc.append(ml.General.compute_multiclass_roc_auc(val_true, val_pred)[0])
           
                
                if os.path.isfile(Train_history_path):       
                    try:
                        #Such model is present and its saved in the same row
                        idx = int(Train_history_Data.index[Train_history_Data['Model_ID']==model].tolist()[0])
                        
                    except:
                        #Such model is not present yet and is saved in the new row
                        idx = len(Train_history_Data['Model_ID'])
                        
                else:
                    print("Data analysis file is not present yet, crating one...")
                    idx = 0  
                    
                row_list =  [model,
                            train_accuracy,
                            train_precision,
                            train_recall,
                            train_f1_score,
                            train_f2_score,
                            train_f0_5_score,
                            train_specificity,
                            train_balanced_accuracy,
                            train_roc_auc,
                            
                            val_accuracy,
                            val_precision,
                            val_recall,
                            val_f1_score,
                            val_f2_score,
                            val_f0_5_score,
                            val_specificity,
                            val_balanced_accuracy,
                            val_roc_auc,
                            ]

                Train_history_Data.loc[idx] = row_list
                
                Train_history_Data.to_feather(Train_history_path)    
             
        #Converting strings to actuall lists in the data
        Train_history_Data = pd.read_feather(Train_history_path)    
        



        train_acc = []
        train_prec = []
        train_recall = []
        train_f1_score = []
        train_specificity = []
        train_balanced_accuracy = []
        train_roc_auc = []
        
        val_acc = []
        val_prec = []
        val_recall = []
        val_f1_score = []
        val_specificity = []
        val_balanced_accuracy = []
        val_roc_auc = []

        
        x_labels = []
        #return Train_history_Data
        for i in range(len(Train_history_Data)):
            train_acc.append(Train_history_Data['train_accuracy'][i])
            val_acc.append(Train_history_Data['val_accuracy'][i])
            
            train_prec.append(Train_history_Data['train_precision'][i])
            val_prec.append(Train_history_Data['val_precision'][i])
            
            train_recall.append(Train_history_Data['train_recall'][i])
            val_recall.append(Train_history_Data['val_recall'][i])
            
            train_f1_score.append(Train_history_Data['train_f1_score'][i])
            val_f1_score.append(Train_history_Data['val_f1_score'][i])
            
            train_specificity.append(Train_history_Data['train_specificity'][i])
            val_specificity.append(Train_history_Data['val_specificity'][i])
            
            train_balanced_accuracy.append(Train_history_Data['train_balanced_accuracy'][i])
            val_balanced_accuracy.append(Train_history_Data['val_balanced_accuracy'][i])
            
            train_roc_auc.append(Train_history_Data['train_roc_auc'][i])
            val_roc_auc.append(Train_history_Data['val_roc_auc'][i])


            #Model plot display name
            idx = int(Data.index[Data['Model_ID']==Train_history_Data["Model_ID"][i]].tolist()[0])
            
            Name_label = Data["Model_WorkName"][idx]
            
            Model_label = Data["Model_Parameters"][idx]
            Model_Dtype = Data["Model_Dtype"][idx]
            if len(str(Model_label)) >3 and len(str(Model_label))<6:
                Model_label = str(round(Model_label/1e3,1))+ "K_"+Model_Dtype
            elif len(str(Model_label)) >=6:
                Model_label = str(round(Model_label/1e6,1))+ "M_"+Model_Dtype
                
            
            Img_H = Data["Img_H"][idx]
            Img_W = Data["Img_W"][idx]
            Img_Dtype = Data["Img_Dtype"][idx]
            Img_Form = Data["Form"][idx]
            
            Img_label = str(Img_H) +"x"+ str(Img_W)+ " "+Img_Form+"_"+Img_Dtype
            
            bs = str(Data["Batch_size"][idx])
            Opt = ast.literal_eval(Data["Optimizer_data"][idx])['name']
            loss = str(Data["Loss_type"][idx])
            Train_label = "Opt:"+Opt+" B_size: "+bs +"\nLoss: "+loss
            
            Epochs_trained = str(Data["Epochs_trained"][idx])
            Epochs_best = str(Data["Epochs_toBest"][idx])
            
            Epochs_label = 'Epochs best/trained: '+Epochs_best+"/"+Epochs_trained
            
            ID_label = "Short ID: "+Data["Model_ID"][idx][0:5]
            
            Plot_label = Name_label+" "+Model_label+"\n"+Img_label+"\n"+Train_label+"\n"+Epochs_label+"\n"+ID_label

            x_labels.append(Plot_label)
           
            
            
            plot_train_merged_metrics = [train_acc,
                                         train_prec,
                                         train_recall,
                                         train_f1_score,
                                         train_specificity,
                                         train_balanced_accuracy,
                                         train_roc_auc]
            
            plot_val_merged_metrics = [val_acc,
                                         val_prec,
                                         val_recall,
                                         val_f1_score,
                                         val_specificity,
                                         val_balanced_accuracy,
                                         val_roc_auc]
            merged_titles =             ["Accuracy",
                                         "Precision",
                                         "Recall",
                                         "F1 Score",
                                         "Specificity",
                                         "Balanced_accuracy",
                                         "ROC_AUC"]
        #return plot_train_merged_metrics     
        for j in range(len(plot_train_merged_metrics)):      
            mn_y =0             
            if merged_titles[j] == "Specificity":
                mn_y = 1-(1/N_classes)
            if merged_titles[j] == "ROC_AUC":
                mn_y = 0.5

            Utils.Analysis_over_train(models_metric = plot_train_merged_metrics[j],
                                val_models_metric = plot_val_merged_metrics[j],
                                x_labels = x_labels,
                                plot_title = merged_titles[j], 
                                show_plots = show_plots,
                                save_plots = save_plots,
                                analysis_folder_path = analysis_folder_directory ,
                                min_y = mn_y
                                )

    
    
    
    def Generate_model_pdf(model_hash,model_params, background_path,file_path):
        def generate_distinct_colors(n):
            """Generate a list of distinct, bright colors."""
            cmap = mpl.colormaps['Paired']  # Use a colormap with many distinct colors
            return [cmap(i) for i in range(n)]
        
        def create_donut_chart(classes, sizes, scale=1.0):
            """
            Create a donut chart with adjustable scale.
            
            Parameters:
            - classes: List of class names.
            - sizes: List of sizes for each class.
            - scale: Scaling factor to adjust the size of the chart.
            """
            fig, ax = plt.subplots(figsize=(10 * scale, 10 * scale))  # Scale the figure size
            total_classes = len(classes)
            # Generate bright and saturated colors
            colors = generate_distinct_colors(len(classes))
            # Adjust font sizes proportionally to the scale, ensuring minimum and maximum limits
            autopct_fontsize = int(round(total_classes*-0.373+scale*12.8+7.4))
            total_classes_fontsize = int(round((total_classes*-0.373+scale*12.8+7.4)*4))  # Ensure max size for total number
            class_name_fontsize = int(round((total_classes*-0.373+scale*12.8+7.4)*1.3))  # Ensure minimum size for class names
        
            
            wedges, texts, autotexts = ax.pie(
                sizes,
                colors=colors,
                startangle=140,
                wedgeprops=dict(width=0.5),  # Adjust width for denser donut
                autopct='%1.1f%%',
                pctdistance=0.7,  # Move percentage text closer to the donut
                textprops=dict(color="black", fontsize=autopct_fontsize)
            )
        
            # Add percentages inside the corresponding segments with smaller font size
            for autotext in autotexts:
                autotext.set_fontsize(autopct_fontsize)  # Reduce percentage font size
                autotext.set_weight('bold')
                autotext.set_color('black')
        
            # Display the total number of classes in the center
            
            ax.text(0, 0, str(total_classes), ha='center', va='center', color='black', fontsize=total_classes_fontsize, weight='bold')
        
            # Add class names in a circular arrangement around the donut
            radius = 1.3  # Distance from the center to place class names
            for i, (wedge, size) in enumerate(zip(wedges, sizes)):
                ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
                y = radius * np.sin(np.deg2rad(ang))
                x = radius * np.cos(np.deg2rad(ang))
                
                # Avoid overlaps by adjusting the position based on angle
                offset = 0.05 * np.sign(x)
                ax.text(x + offset, y, classes[i], ha='center', va='center', color='black', fontsize=class_name_fontsize, weight='bold')
        
            # Remove axes and set transparent background
            ax.axis('equal')
            plt.axis('off')
            fig.patch.set_facecolor('none')  # Set figure background to transparent
        
            # Save plot as SVG to a bytes buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='svg', bbox_inches='tight', transparent=True)
            plt.close(fig)
            buffer.seek(0)
            
            # Create a drawing object from the SVG buffer
            drawing = svg2rlg(buffer)
            
            return drawing
            
        def find_number_after_m(s):
            # Use regex to find 'm' followed by one or more digits
            match = re.search(r'm(\d+)', s)
            if match:
                return int(match.group(1))
            else:
                return None
        
        
        def sort_dict(d):
            # Check if "learning_rate" is in the dictionary
            has_learning_rate = "learning_rate" in d
            
            # Separate the keys
            sorted_keys = sorted(d.keys())
            
            if has_learning_rate:
                # Remove "learning_rate" and place it at the beginning
                sorted_keys.remove("learning_rate")
                sorted_keys.insert(0, "learning_rate")
            
            # Create a new sorted dictionary
            sorted_dict = {key: d[key] for key in sorted_keys}
            
            return sorted_dict
        
        def split_string_at_length(s, max_length):
            if len(s) <= max_length:
                return s
            
            # Find the best split point
            split_point = max_length+1
            if ' ' in s[max_length:]:
                # Look for the nearest space after max_length to avoid splitting words
                split_point = s.rfind(' ', 0, max_length)
                if split_point == -1:
                    split_point = max_length
            else:
                # No spaces found, use the middle point
                split_point = len(s) // 2
        
            return s[:split_point].rstrip() + '\n' + s[split_point:].lstrip()
        
        
        
        model_work_name = model_params["User Architecture Name"]
        total_params = model_params["Total Parameters"]
        layer_number = model_params["Number of Layers"]
        model_dataType = model_params["Model Datatype"]
        loss_function = model_params["Loss function"]
        batch_size = model_params["Batch Size"]
        
        optimizer_dict = ast.literal_eval(model_params["Optimizer Parameters"])
        
        optimizer_name = optimizer_dict["name"]
        del optimizer_dict["name"]
        
        optimizer_dict = sort_dict(optimizer_dict)
        
        img_h = model_params["Image Height"]
        img_w = model_params["Image Width"]
        img_color = "Grayscale" if model_params["Grayscale"] else 'RGB'
        img_dataType = model_params["Image Datatype"]
        
        dataset_size = sum(c[1] for c in model_params["Class Size"])
        augm = find_number_after_m(model_params["Augmentation Mark"])
        
        real_data_part = round(dataset_size*1 / augm/dataset_size,3)
        aug_data_part = 1-real_data_part
        
        test_split = model_params["Test split"]
        val_split = model_params["Validation split"]
        
        classes = [c[0] for c in model_params["Class Size"]]
        class_size  = [c[1] for c in model_params["Class Size"]]
        
        checkpoint_monitor = model_params["Checkpoint monitor"]
        checkpoint_mode = model_params["Checkpoint mode"]
        ###
        total_params = str(total_params)
        layer_number = str(layer_number)
        batch_size = str(batch_size)
        img_h = str(img_h)
        img_w = str(img_w)
        dataset_size = str(dataset_size)
        
        classes = [s.replace(".npy","") for s in classes]
        classes = [s.replace(".csv","") for s in classes]
        classes = [split_string_at_length(s, 8) for s in classes]
        
        r = 0.25
        g = 0.25
        b = 0.25
        
        # Register custom font
        pdfmetrics.registerFont(TTFont('Arial-Black', 'ariblk.ttf'))
        
        # Create a PDF document
        pdf = canvas.Canvas(file_path, pagesize=letter)
        width, height = letter
        
        # Draw the background image
        pdf.drawImage(background_path, 0, 0, width=width, height=height)
        
        # Set the transparency for the background image (if needed)
        pdf.setFillColorRGB(1, 1, 1, alpha=0.1)
        pdf.rect(0, 0, width, height, stroke=0, fill=1)
        
        
        #Title section
        pdf.setFont("Helvetica", 16.5)
        pdf.setFillColor(colors.black)  
        pdf.drawString(30, height - 50, model_hash)
        
        
        supt_size = 13
        #Model info section
        pdf.setFont("Arial-Black", 20)
        pdf.setFillColor(colors.black)  
        pdf.drawString(30, height - 100, "Model Info")
        
        pdf.setFont("Arial-Black", supt_size)
        pdf.setFillColor(colors.black)  
        pdf.drawString(30, height - 120, "Model workName: ")
        
        pdf.setFont("Helvetica", supt_size)
        pdf.setFillColorRGB(r,g,b)  
        pdf.drawString(170, height - 120, model_work_name)
        
        pdf.setFont("Arial-Black", supt_size)
        pdf.setFillColor(colors.black)  
        pdf.drawString(30, height - 140, "Total params: ")
        
        pdf.setFont("Helvetica", supt_size)
        pdf.setFillColorRGB(r,g,b)  
        pdf.drawString(170, height - 140, total_params)
        
        pdf.setFont("Arial-Black", supt_size)
        pdf.setFillColor(colors.black)  
        pdf.drawString(30, height - 160, "Number of layers: ")
        
        pdf.setFont("Helvetica", supt_size)
        pdf.setFillColorRGB(r,g,b)  
        pdf.drawString(170, height - 160, layer_number)
        
        pdf.setFont("Arial-Black", supt_size)
        pdf.setFillColor(colors.black)  
        pdf.drawString(30, height - 180, "Model dataType: ")
        
        pdf.setFont("Helvetica", supt_size)
        pdf.setFillColorRGB(r,g,b)  
        pdf.drawString(170, height - 180, model_dataType)
        
        pdf.setFont("Arial-Black", supt_size)
        pdf.setFillColor(colors.black)  
        pdf.drawString(30, height - 200, "Loss: ")
        
        pdf.setFont("Helvetica", supt_size)
        pdf.setFillColorRGB(r,g,b)  
        pdf.drawString(170, height - 200, loss_function)
        
        pdf.setFont("Arial-Black", supt_size)
        pdf.setFillColor(colors.black)  
        pdf.drawString(30, height - 220, "Batch_size: ")
        
        pdf.setFont("Helvetica", supt_size)
        pdf.setFillColorRGB(r,g,b)  
        pdf.drawString(170, height - 220, batch_size)
        
        
        #######################################################
        supt_size = 13
        #Optimizer info section
        pdf.setFont("Arial-Black", 20)
        pdf.setFillColor(colors.black)  
        pdf.drawString(30, height - 280, "Optimizer: ")
        
        pdf.setFont("Arial-Black", 20)
        pdf.setFillColorRGB(r,g,b)  
        pdf.drawString(170, height - 280, optimizer_name)
        
        for i,item in enumerate(optimizer_dict):
            offset = i*20
            h = 300+offset
            pdf.setFont("Arial-Black", supt_size)
            pdf.setFillColor(colors.black)  
            suptitle = item[0].upper()+item[1:].lower()
            pdf.drawString(30, height - h, suptitle+": ")
        
            pdf.setFont("Helvetica", supt_size)
            pdf.setFillColorRGB(r,g,b)  
            pdf.drawString(170, height - h, str(optimizer_dict[item]))
            
        #To unmess up
        ###########
        pdf.setFont("Arial-Black", supt_size)
        pdf.setFillColor(colors.black)  
        pdf.drawString(30, height - 500, "Checkp. monitor: ")
        
        pdf.setFont("Helvetica", supt_size)
        pdf.setFillColorRGB(r,g,b)  
        pdf.drawString(170, height - 500, checkpoint_monitor)
        
        
        pdf.setFont("Arial-Black", supt_size)
        pdf.setFillColor(colors.black)  
        pdf.drawString(30, height - 520, "Checkp. mode: ")
        
        pdf.setFont("Helvetica", supt_size)
        pdf.setFillColorRGB(r,g,b)  
        pdf.drawString(170, height - 520, checkpoint_mode)
        ###########
        ###########
        
        
        
        column_2 = 340
        column_2_space = 180
        
        supt_size = 13
        #Data info section
        pdf.setFont("Arial-Black", 20)
        pdf.setFillColor(colors.black)  
        pdf.drawString(column_2, height - 100, "Data parameters: ")
        
        pdf.setFont("Arial-Black", supt_size)
        pdf.setFillColor(colors.black)  
        pdf.drawString(column_2, height - 120, "Image height: ")
        
        pdf.setFont("Helvetica", supt_size)
        pdf.setFillColorRGB(r,g,b)  
        pdf.drawString(column_2+column_2_space, height - 120, img_h)
        
        pdf.setFont("Arial-Black", supt_size)
        pdf.setFillColor(colors.black)  
        pdf.drawString(column_2, height - 140, "Image width: ")
        
        pdf.setFont("Helvetica", supt_size)
        pdf.setFillColorRGB(r,g,b)  
        pdf.drawString(column_2+column_2_space, height - 140, img_w)
        
        pdf.setFont("Arial-Black", supt_size)
        pdf.setFillColor(colors.black)  
        pdf.drawString(column_2, height - 160, "Image color: ")
        
        pdf.setFont("Helvetica", supt_size)
        pdf.setFillColorRGB(r,g,b)  
        pdf.drawString(column_2+column_2_space, height - 160, img_color)
        
        pdf.setFont("Arial-Black", supt_size)
        pdf.setFillColor(colors.black)  
        pdf.drawString(column_2, height - 180, "Image dataType: ")
        
        pdf.setFont("Helvetica", supt_size)
        pdf.setFillColorRGB(r,g,b)  
        pdf.drawString(column_2+column_2_space, height - 180, img_dataType)
        
        
        supt_size = 13
        #Data distribution section
        pdf.setFont("Arial-Black", 20)
        pdf.setFillColor(colors.black)  
        pdf.drawString(column_2, height - 280, "Data distribution: ")
        
        pdf.setFont("Arial-Black", supt_size)
        pdf.setFillColor(colors.black)  
        pdf.drawString(column_2, height - 300, "Total dataSet size: ")
        
        pdf.setFont("Helvetica", supt_size)
        pdf.setFillColorRGB(r,g,b)  
        pdf.drawString(column_2+column_2_space, height - 300, dataset_size)
        
        pdf.setFont("Arial-Black", supt_size)
        pdf.setFillColor(colors.black)  
        pdf.drawString(column_2, height - 320, "Unmodified/Augmented: ")
        
        pdf.setFont("Helvetica", supt_size)
        pdf.setFillColorRGB(r,g,b)  
        pdf.drawString(column_2+column_2_space, height - 320, dataset_size+"/"+str(int(dataset_size)*(augm-1)))
        
        
        #Real Aug plot section
        ######################################################
        bar_width = 245
        bar_height = 26
        contour = 2
        
        image = Image.new('RGB', (bar_width, bar_height), 'black')
        draw = ImageDraw.Draw(image)
         # Draw the background square with black contour
        augm_box = [contour+1, contour+1, bar_width-(2*contour), bar_height-(2*contour)]
        draw.rectangle(augm_box, fill='red')
         # Draw the real data part
        real_box= [contour+1, contour+1, (bar_width-(2*contour))*real_data_part, bar_height-(2*contour)]
        draw.rectangle(real_box, fill='green')
        
        real_middle = (bar_width*real_data_part/2)+column_2
        aug_middle = bar_width*(real_data_part+(aug_data_part*0.5))+column_2
        
        # Save image to a buffer
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        image = ImageReader(buffer)
        pdf.drawImage(image, column_2, height - 400)
        
        
        # Save and draw the image
        #image.save("aug_plot.png")
        #pdf.drawImage("aug_plot.png", column_2, height - 400)
        
        
        #Real Aug plot text section
        ######################################################
        def add_word_with_pointers(pdf, name, variable, pointer_x,pointer_y,horizontal,vertical,left_border,right_border):
        
            # Calculate the width of the word
            word_width_1 = stringWidth(name, "Arial-Black", 12)
            word_width_2 = stringWidth(name, "Helvetica", 12)
            string_width = word_width_1+word_width_2+1
            
            if horizontal == "left":
                word_x = pointer_x - string_width - 15 
            elif horizontal == "right":
                word_x = pointer_x + 15
            else:
                print("Horizontal should be 'right' or 'left'")
                return
            
            if vertical == "up":
                word_y = pointer_y + 40
            elif vertical == "down":
                word_y = pointer_y - 40+26
            else:
                print("Vertical should be 'up' or 'down'" )
                return
            
            if word_x < left_border:
                word_x = left_border
            if word_x+string_width> right_border:
                word_x = right_border-string_width
            
            
            #Writing words
            pdf.setFont("Arial-Black", 12)
            pdf.setFillColor(colors.black)  
            pdf.drawString(word_x, word_y , name)
        
            pdf.setFont("Helvetica", 12)
            pdf.setFillColor(colors.black)  
            pdf.drawString(word_x+word_width_2+2, word_y , variable)
            
            # Draw the underline
            underline_start = (word_x, word_y - 5)
            underline_end = (word_x + string_width-5, word_y - 5)
            pdf.setLineWidth(2)
            pdf.line(underline_start[0], underline_start[1], underline_end[0], underline_end[1])
            
        
           # Determine the closest edge for the pointer line
            if abs(pointer_x - underline_start[0]) < abs(pointer_x - underline_end[0]):
                pointer_start_x = underline_start[0]
            else:
                pointer_start_x = underline_end[0]
        
            pointer_start_y = underline_start[1]
            
            if vertical == "up":
                pointer_y = pointer_y + 26
            # Draw the pointer line
            pdf.line(pointer_start_x, pointer_start_y, pointer_x, pointer_y)
        
            
        if aug_data_part == 0:
            h = "right"
        else:
            if real_data_part / aug_data_part > 0.5:
                h = "left"
            else:
                h = "right"
        if aug_data_part ==0.5:
            real_middle = real_middle-30
            h = "right"
            
            
        add_word_with_pointers(pdf = pdf,
                               name = "Real:  ",
                               variable = str(int(round(real_data_part*100,2)))+"%",
                               pointer_x = real_middle,
                               pointer_y = 391,
                               horizontal = h,
                               vertical = "up",
                               left_border = column_2,
                               right_border = column_2+bar_width
                               )
        
        
        
        if aug_data_part>0:
            if aug_data_part!=0.5:
                if real_data_part / aug_data_part > 0.5:
                    h = "left"
                else:
                    h = "right"
            else:
                h = "right"
                aug_middle = aug_middle-30
            add_word_with_pointers(pdf = pdf,
                                   name = "Aug:  ",
                                   variable = str(int(round(aug_data_part*100,2)))+"%",
                                   pointer_x = aug_middle,
                                   pointer_y = 391,
                                   horizontal = h,
                                   vertical = "up",
                                   left_border = column_2,
                                   right_border = column_2+bar_width
                                   )
        

        image = Image.new('RGB', (bar_width, bar_height), 'black')
        draw = ImageDraw.Draw(image)
        #Draw test part
        test_box = [contour+1, contour+1, bar_width-(2*contour), bar_height-(2*contour)]
        draw.rectangle(test_box, fill='red')
        # Draw the val data part
        val_box= [contour+1, contour+1, (bar_width-(2*contour))*(1-test_split), bar_height-(2*contour)]
        draw.rectangle(val_box, fill='blue')
        # Draw the real data part
        train_box= [contour+1, contour+1, (bar_width-(2*contour))*(1-test_split-val_split), bar_height-(2*contour)]
        draw.rectangle(train_box, fill='green')
        # Save image to a buffer
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        image = ImageReader(buffer)
        
        pdf.drawImage(image, column_2, 320)
        
        train_middle = (bar_width*(1-val_split-test_split))/2+column_2
        val_middle = bar_width*(1-val_split-test_split +val_split/2)+column_2
        test_middle =bar_width*(1-test_split/2)+column_2
        
        
        
        if val_split>0:
            train_split = 1-val_split-test_split
            if train_split/val_split>1.3:
                h = "left"
            else:
                h = "right"
            if val_split>train_split:
                h = "right"
                
        add_word_with_pointers(pdf = pdf,
                               name = "Train:  ",
                               variable = str(int(round((1-val_split-test_split)*100,2)))+"%",
                               pointer_x = train_middle,
                               pointer_y = 319,
                               horizontal = h,
                               vertical = "up",
                               left_border = column_2,
                               right_border = column_2+bar_width
                               )
        
        if val_split>0:
            train_split = 1-val_split-test_split
         
            if train_split/val_split>1.3:
                h = "left"
            else:
                h = "right"
                
            if test_split:
                if test_split>val_split:
                    h = "right"
            add_word_with_pointers(pdf = pdf,
                                   name = "Val:  ",
                                   variable = str(int(round(val_split*100,2)))+"%",
                                   pointer_x = val_middle,
                                   pointer_y = 319,
                                   horizontal = h,
                                   vertical = "up",
                                   left_border = column_2,
                                   right_border = column_2+bar_width
                                   )
            
        if test_split>0:
            add_word_with_pointers(pdf = pdf,
                                   name = "Test:  ",
                                   variable = str(int(round(test_split*100,2)))+"%",
                                   pointer_x = test_middle,
                                   pointer_y = 321,
                                   horizontal = "left",
                                   vertical = "down",
                                   left_border = column_2,
                                   right_border = column_2+bar_width
                                   )
        
        donut_plot = create_donut_chart(classes =classes , sizes = class_size, scale = 0.3)
        renderPDF.draw(donut_plot, pdf, 280, 30)
        
        
        #Add small watermark
        pdf.setFont("Helvetica", 8)
        pdf.setFillColorRGB(0.5,0.5,0.5)  
        pdf.drawString(470,7, "Report generated by NeuroUtils library")
        
        
        pdf.save()
    
    
    

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
                   
    
    def Process_Data(x , y ,dataset_multiplier, DataProcessed_directory, val_split = 0.2, test_split = 0, flipRotate = False , randBright = False , gaussian = False , denoise = False , contour = False ):        
        if val_split+test_split>1:
            raise ValueError("Val_split and test_split cannot sum above 1, lower your values so they do not exceed 1")
        elif val_split+test_split>0.5:
            print("Size of train dataset is set under 50% of total amount of data, you may consider lowering values of validation and test set to achieve better performance")
        
        if test_split == 0:
            Create_test_set = False
        else:
            Create_test_set = True
            
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

        f1_factor = val_split+test_split
        f2_factor = test_split/f1_factor

        if not Create_test_set:
            x_train , x_val , y_train , y_val = train_test_split(x,y,test_size = f1_factor ,stratify = y, shuffle = True)
        else:
            x_train , x_val , y_train , y_val = train_test_split(x,y,test_size = f1_factor ,stratify = y, shuffle = True)
            x_val , x_test , y_val , y_test = train_test_split(x_val,y_val,test_size = f2_factor ,stratify = y_val, shuffle = True)
        
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
        def __init__(self, filepath, best_filepath, monitor='val_loss', mode='min', min_delta=0):
            super().__init__()
            self.filepath = filepath
            self.best_filepath = best_filepath
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
            self.model.save(self.filepath)

            if current_score is not None:
                if self._is_improvement(current_score, self.best_score):
                    print(f"\nImprovement detected in {self.monitor}. Saving model with score: {current_score:.4f}")
                    self.best_score = current_score
                    self.model.save(self.best_filepath)
                    self._save_best_score(self.best_score)
                else:
                    print(f"\nNo improvement in {self.monitor}. Not saving model. Model score: {current_score:.4f} vs {self.best_score:.4f}")


        def _is_improvement(self, current, best):
            if self.mode == 'min':
                return current < best - self.min_delta
            else:
                return current > best + self.min_delta
            
    class MetricsCallback(tf.keras.callbacks.Callback):
        def __init__(self, file_path='Metrics.csv', train_data=None, validation_data=None):
            super().__init__()
            self.file_path = file_path
            self.train = train_data
            self.val = validation_data
    
            # Create the file and write the headers if it doesn't exist
            if not os.path.exists(self.file_path):
                with open(self.file_path, 'w') as f:
                    f.write('epoch,train_true,train_pred,val_true,val_pred\n')
    
        def on_epoch_end(self, epoch, logs=None):
            train_metrics = [None] * 4
            val_metrics = [None] * 4
    
            if self.train is not None:
                print("Calculating metrics for train set...")
                y_pred = self.model.predict(self.train[0])
                y_pred = np.argmax(y_pred,axis = 1).astype(int).tolist()
                
                y_true = np.argmax(self.train[1],axis = 1).astype(int).tolist()
                
                train_metrics = [y_true,y_pred]
                del y_pred
                del y_true
                tf.keras.backend.clear_session()
                gc.collect()
    
            if self.val is not None:
                print("Calculating metrics for validation set...")
                y_pred = self.model.predict(self.val[0])
                y_pred = np.argmax(y_pred,axis = 1).astype(int).tolist()
                
                y_true = np.argmax(self.val[1],axis = 1).astype(int).tolist()
                
                val_metrics = [y_true,y_pred]
                del y_pred
                del y_true
                tf.keras.backend.clear_session()
                gc.collect()
    
            # Append the metrics for this epoch to the CSV file
            with open(self.file_path, mode='a') as f:
                f.write(f'{epoch},"{str(train_metrics[0])}","{str(train_metrics[1])}",'
                        f'"{str(val_metrics[0])}","{str(val_metrics[1])}"\n')
    
            print("-----------------------------------------------------------------------")
            print("\n")
    

    class SilentProgbarLogger(tf.keras.callbacks.ProgbarLogger):
        def on_epoch_end(self, epoch, logs=None):
            pass  # Override to prevent the last message from being printed


                
    def Initialize_weights_and_training(x_train, y_train, model, model_directory, train, epochs, patience, batch_size,min_delta, monitor, mode, x_val=None, y_val=None, device = "CPU:0"):    
        #!!! Model training
        #########################################################################
        #########################################################################
        #Check if directory of trained model is present, if not, create one 
        if not os.path.isdir(model_directory):
            os.makedirs(model_directory)
            print("Creating model directory storage directory...\n")
            
        model_weights_directory = os.path.join(model_directory , "Model.keras")
        model_history_directory = os.path.join(model_directory , "Model_history.csv")
        model_metrics_directory = os.path.join(model_directory , "Model_metrics.csv")
        best_model_directory = os.path.join(model_directory , "Model_best.keras")
        
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
                                                         monitor=monitor,
                                                         min_delta=min_delta),
                        #Checkpoint model if performance is increased
                        Utils.SaveBestModel(filepath=model_weights_directory, best_filepath = best_model_directory, monitor=monitor, mode=mode,min_delta = min_delta),       
                        #Save data through training
                        tf.keras.callbacks.CSVLogger(filename = model_history_directory , append = csv_append),
                        #Saving model metrics TP,FP,FN,TN
                        Utils.SilentProgbarLogger(count_mode = 'steps'),
                        Utils.MetricsCallback(file_path = model_metrics_directory,train_data = (x_train,y_train),validation_data = (x_val,y_val))
                        ]

            with tf.device(device):
                
                #Start measuring time
                timer_start = timer()
                model.fit(x_train,y_train,
                          initial_epoch = starting_epoch,
                          validation_data = (x_val , y_val),
                          epochs=epochs,
                          batch_size = batch_size,
                          callbacks = callbacks,
                          use_multiprocessing = True,
                          verbose = 1
                          )
                
                print("Time took to train model: ",round(timer()-timer_start),2)    
                
            
            #Save the best achieved model
            print("Loading model which was performing best during training...\n")
            model.load_weights(best_model_directory)   
                
        
             
         
            
         
        #########################################################################
        #########################################################################
        return model
    
    

       
    def Initialize_Results(model,model_directory, dictionary,evaluate, x_train = None ,y_train = None ,x_val = None , y_val = None , x_test = None , y_test = None,show_plots = False,save_plots = True):    
        #!!! Model results
        #########################################################################
        #########################################################################
        
        #Plot model training history
        model_history_directory = os.path.join(model_directory , "Model_history.csv")
        Model_history = pd.read_csv(model_history_directory)
        ml.General.Model_training_history_plot_CSV(Model_history)
        if save_plots:
            file_name = "Overall train history.png"
            plot_path = os.path.join(model_directory,file_name)
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)

        if not show_plots:
            plt.close()
        
        try:
            #Create confusion matrix
            #Predict classes
            print("\nPredicting classes based on train set...")
            plot_title = "Train"
            y_pred = model.predict(x_train)
            
            plt.figure()
            ml.General.Conf_matrix_classification(y_train ,y_pred , dictionary, plot_title, normalize = True)
            
            if save_plots:
                file_name = "Conf_matrix "+plot_title + ".png"
                plot_path = os.path.join(model_directory,file_name)
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)

            if not show_plots:
                plt.close()
                
        except:
            print("No train set provided, skipping...")
                   
        try:
            #Create confusion matrix
            #Predict classes
            print("\nPredicting classes based on validation set...")
            plot_title = "Validation"
            y_pred = model.predict(x_val)
            
            plt.figure()
            ml.General.Conf_matrix_classification(y_val ,y_pred , dictionary, plot_title, normalize = True)
            
            if save_plots:
                file_name = "Conf_matrix "+plot_title + ".png"
                plot_path = os.path.join(model_directory,file_name)
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)

            if not show_plots:
                plt.close()
        except:
            print("No validation set provided, skipping...")    
            
        try:
            #Create confusion matrix
            #Predict classes
            print("\nPredicting classes based on test set...")
            plot_title = "Test"
            y_pred = model.predict(x_test)
            
            plt.figure()
            ml.General.Conf_matrix_classification(y_test ,y_pred , dictionary, plot_title, normalize = True)
            
            if save_plots:
                file_name = "Conf_matrix "+plot_title + ".png"
                plot_path = os.path.join(model_directory,file_name)
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)

            if not show_plots:
                plt.close()
        except:
            print("No test set provided, skipping...")
        
    
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
        def  __init__(self,Database_Directory):
            self.PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(sys.argv[0]))
            self.DATABASE_DIRECTORY = Database_Directory
            
            """

            #Model
            

            """
            
            
            
            
        def __str__(self):
            return "NO DESCRIPTION"
            
    
        def Initialize_data(self,Img_Height,Img_Width,Grayscale = False,CSV_Load = False): 
            self.IMG_H = Img_Height
            self.IMG_W = Img_Width
            self.GRAYSCALE = Grayscale
            
            self.CSV_LOAD = CSV_Load
            
            self.FORM = "Grayscale" if self.GRAYSCALE else "RGB"
            self.CHANNELS = 3 if self.FORM == "RGB" else 1
            
            self.DATA_DIRECTORY = os.path.join(self.PROJECT_DIRECTORY , "DataSet" , str(str(self.IMG_H)+"x"+str(self.IMG_W)+"_"+self.FORM))
            
            """Initializing dataset from main database folder with photos to project folder in numpy format. Photos are 
            Resized and cropped without loosing much aspect ratio, r parameter decides above what proportions of edges 
            image will be cropped to square instead of squeezed""" 
            
            Utils.Initialize_data(self.DATABASE_DIRECTORY, self.DATA_DIRECTORY, self.IMG_H, self.IMG_W, self.GRAYSCALE , self.CSV_LOAD)
            ########################################################
        def Load_and_merge_data(self,Reduced_class_size = None):
            """Loading dataset to memory from data directory in project folder, sets can be reduced to equal size
            to eliminate disproportions if they are not same size at the main database
            In this module dictionary with names of classes is created as well, names are based on names of datsets
            Datasets names are based on the folder names in main database folder"""
            
            self.REDUCED_SET_SIZE = Reduced_class_size
            
            X, Y, DICTIONARY = ml.DataSets.Load_And_Merge_DataSet(self.DATA_DIRECTORY , self.REDUCED_SET_SIZE )
            
            return X, Y, DICTIONARY
            ########################################################
            
        def Process_data(self,X,Y,Val_split,Test_split,DataSet_multiplier = 1,DataType = "float32",FlipRotate = False,
                                                                                                      RandBright = False,
                                                                                                      Gaussian_noise = False,
                                                                                                      Denoise = False,
                                                                                                      Contour = False):
            
            self.VAL_SPLIT = Val_split
            self.TEST_SPLIT = Test_split
            self.DATASET_MULTIPLIER = DataSet_multiplier
            self.DATA_TYPE = DataType
            
            self.FLIPROTATE = FlipRotate
            self.RANDBRIGHT = RandBright
            self.GAUSSIAN = Gaussian_noise
            self.DENOISE = Denoise
            self.CONTOUR = Contour
            
            self.CHANNELS = self.CHANNELS+1 if self.CONTOUR else self.CHANNELS
            
            cr = 0 if self.REDUCED_SET_SIZE is None else self.REDUCED_SET_SIZE
            self.PARAM_MARK = "_m"+str(self.DATASET_MULTIPLIER)+"_cr"+str(cr)+"_"+ "_".join(["1" if x else "0" for x in [self.FLIPROTATE, self.RANDBRIGHT, self.GAUSSIAN, self.DENOISE, self.CONTOUR]])
            self.DATAPROCESSED_DIRECTORY = os.path.join(self.PROJECT_DIRECTORY , "DataSet_Processed" , str(str(self.IMG_H)+"x"+str(self.IMG_W)+"_"+self.FORM),self.PARAM_MARK)
            if self.TEST_SPLIT == 0:
                create_test_set = False
            else:
                create_test_set = True
            #3
            ########################################################
            self.SPLIT_MARK = "Val_"+str(round(Val_split,3))+"  Test_"+str(round(Test_split,3))
            self.DATAPROCESSED_DIRECTORY = os.path.join(self.DATAPROCESSED_DIRECTORY,self.SPLIT_MARK)
            if not create_test_set:
                X_TRAIN , Y_TRAIN, X_VAL , Y_VAL = Utils.Process_Data(X, Y, self.DATASET_MULTIPLIER, self.DATAPROCESSED_DIRECTORY, self.VAL_SPLIT, self.TEST_SPLIT, self.FLIPROTATE, self.RANDBRIGHT, self.GAUSSIAN, self.DENOISE, self.CONTOUR)
            
            else:
                X_TRAIN , Y_TRAIN, X_VAL , Y_VAL , X_TEST , Y_TEST = Utils.Process_Data(X, Y, self.DATASET_MULTIPLIER, self.DATAPROCESSED_DIRECTORY, self.VAL_SPLIT, self.TEST_SPLIT, self.FLIPROTATE, self.RANDBRIGHT, self.GAUSSIAN, self.DENOISE, self.CONTOUR)
            
            if create_test_set:
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
            
            
            
    
    
        def Initialize_weights_and_training(self, Model, Architecture_name, Epochs, Batch_size, Train = True, Patience = 10, Min_delta_to_save = 0.1, Checkpoint_monitor = "val_loss", Checkpoint_mode = "min", Device = "CPU", add_config_info = None):
            self.ARCHITECTURE_NAME = Architecture_name
            self.EPOCHS = Epochs
            self.BATCH_SIZE = Batch_size
            self.CHECKPOINT_MONITOR = Checkpoint_monitor
            self.CHECKPOINT_MODE = Checkpoint_mode
            
            #5
            ########################################################
            #Create data for parameters file
            n_classes = self.N_CLASSES
            class_size = self.CLASS_OCCURENCE
            val_split = self.VAL_SPLIT
            test_split = self.TEST_SPLIT
            
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
                "Validation split": val_split,
                "Test split": test_split,
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
                "Checkpoint monitor": self.CHECKPOINT_MONITOR,
                "Checkpoint mode": self.CHECKPOINT_MODE,
                "Additional information": add_config_info
            }
            
            f_name = ml.General.hash_string(str(content))
            model_directory = os.path.join("Models_saved", str(f_name))
            self.MODEL_DIRECTORY = model_directory
            
            if not os.path.exists(model_directory):
                print("Model Created")
                os.mkdir(model_directory)
            
            params_directory = os.path.join(model_directory,"Model_parameters.json")
            model_png_directory = os.path.join(model_directory,"Model_architecture_view.png")
            model_json_directory = os.path.join(model_directory,"Model_architecture_json.json")
            # Write dictionary to JSON file
            with open(params_directory, 'w') as json_file:
                json.dump(content, json_file, indent=4)
               
            if not os.path.exists(model_png_directory): 
                #Generate model png view of architecture
                tf.keras.utils.plot_model(model = Model,
                                          to_file = model_png_directory,
                                          show_shapes = True,
                                          show_layer_names = True
                                          )
                print("Model architecture png view saved")
            if not os.path.exists(model_json_directory):
                #Save model architecture only so it can be retrieved in future
                ml.General.save_model_as_json(model = Model, filename = model_json_directory)
                print("Model architecture JSON template saved")
            
            #Create and save pdf of model
            Utils.Generate_model_pdf(model_hash = f_name,
                                     model_params = content,
                                     background_path = r"C:\Users\Stacja_Robocza\Desktop\NeuroUtils\Assets\Background.png",
                                     file_path = os.path.join(self.MODEL_DIRECTORY , "Model_preview.pdf")
                                     )


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
                                                       monitor = self.CHECKPOINT_MONITOR,
                                                       mode = Checkpoint_mode,
                                                       device = Device
                                                       )
            
            return self.MODEL
            ########################################################
    
        def Initialize_results(self,show_plots = False,save_plots = True,Evaluate = False):
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
                                  self.Y_TEST,
                                  show_plots,
                                  save_plots
                                  )
            ######################################################## 
            

       
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
            
        
            
                        
                    
            
            
            
            
            
            
            
            
            
            
            
            
            
            

            
