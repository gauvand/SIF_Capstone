"""
<center><h1>UVA Psychology Capstone   </h1></center>
<center><h1>Brain Computer Interface with Scale Invariant Temporal History</h1></center>

A preliminary Deep_iSith model that trains on EEG data.
Gaurav Anand, Arshiya Ansari, Beverly Dobrenz, Yibo Wang  
data source: Grasp-and-Lift EEG  
https://www.kaggle.com/c/grasp-and-lift-eeg-detection/data  
For now, only consider one subject and one trial at a time.
Predict only one event/channel a time (since there are events overlapping), and incorporate sliding-window standardization and filtering 
The code is tested on Rivanna with GPU. (may needs some work with CPU only)

"""

# preprocessing
import mne
import numpy as np
import math
import pandas as pd
from mne.io import RawArray
from mne.channels import make_standard_montage
from mne.epochs import concatenate_epochs
from mne import create_info, find_events, Epochs, concatenate_raws, pick_types
from mne.decoding import CSP

# pytorch
import torch
import torch.nn
import torch.nn.functional as F
ttype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
labeltype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
print(ttype)
from torch.utils.data import Dataset,DataLoader 

# deep_iSITH is being used here, not deep_sith
#from sith import iSITH
#from sith import DeepSITH

import matplotlib.pyplot as plt


# training 
from torch import nn as nn
from math import factorial
import random
import seaborn as sn
import os 
from os.path import join
import glob

# validation
from sklearn.metrics import roc_curve, auc, roc_auc_score, matthews_corrcoef,confusion_matrix,plot_roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score

from tqdm.notebook import tqdm
import pickle
import datetime

sn.set_context("poster")



def evaluation(model, val_loader):
    """
    Test for accuracy
    Iterate through each batch and make prediciton and calculate performance metrics
    Use **matthews correlation coeefficient** since the data are imbanlanced
    Again 
    Signals need to be in correct format. validation input: [nbatch x 1 x nFeutures x time] tensor.

    The target has dimension of [time] tensor, in which each entry should be one of the numbers in 
    {0,1,2, ... K} at any time point.  
    
    """
    auc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for _, (val_x, labels) in enumerate(val_loader):
        out_val = model(val_x)
        #print(out_val.shape)
        # pass through a softmax to tansform to probability on the third dimention (nbatch, seq, outFeature)
        res = torch.nn.functional.softmax(out_val, dim=2)
        #print(res.shape)
        # predict should also be the second dimension [1] to clauclate AUC
        y_pred = res[:,:,1]

        # flatten the predicted result 
        y_score = np.ndarray.flatten(y_pred.detach().cpu().numpy())

        # flatten the predicted result 
        y_true = np.ndarray.flatten(labels.detach().cpu().numpy())
        #print(y_true)
        try: # in case there is only one class
            auc = roc_auc_score(y_true = y_true,y_score = y_score)
            #print(auc)
            precision = precision_score(y_true = y_true,y_score = y_score)
            recall = recall_score(y_true = y_true,y_score = y_score)
            f1 = f1_score
            auc_list.append(auc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
        except:
            auc_list.append(np.nan)
            precision_list.append(np.nan)
            recall_list.append(np.nan)
            f1_list.append(np.nan)
    auc_acc = np.array(auc_list)
    print(auc_acc)
    p_acc = np.array(auc_list)
    r_acc = np.array(auc_list)
    f1_acc = np.array(auc_list)
        #acc = matthews_corrcoef(y_true = y_true, y_pred = y_pred)
        # return the average
    return np.nanmean(auc_acc),np.nanmean(p_acc),np.nanmean(r_acc),np.nanmean(f1_acc)



