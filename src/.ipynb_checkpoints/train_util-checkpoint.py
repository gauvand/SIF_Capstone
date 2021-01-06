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


from tqdm.notebook import tqdm
import pickle
import datetime

sn.set_context("poster")

def creat_mne_raw_object(fname,read_events=True):
    """
    obtained from @author: alexandrebarachant
    https://www.kaggle.com/alexandrebarachant/beat-the-benchmark-0-67
    Create a mne raw instance from csv file.
    Make sure the events and data files are in the same folder
    data are transformed into microvolts
    """
    # Read EEG file
    data = pd.read_csv(fname)
    
    # get chanel names
    ch_names = list(data.columns[1:])
    
    # read EEG standard montage from mne
    #montage = make_standard_montage('standard_1005')

    ch_type = ['eeg']*len(ch_names)
    data = 1e-6*np.array(data[ch_names]).T
    
    if read_events:
        # events file
        ev_fname = fname.replace('_data','_events')
        # read event file
        events = pd.read_csv(ev_fname)
        events_names = events.columns[1:]
        events_data = np.array(events[events_names]).T
        # define channel type, the first is EEG, the last 6 are stimulations
        ch_type.extend(['stim']*6)
        ch_names.extend(events_names)
        # concatenate event file and data
        data = np.concatenate((data,events_data))
        
    # create and populate MNE info structure
    info = create_info(ch_names,sfreq=500.0, ch_types=ch_type)
    #info['filename'] = fname
    #print(info)
    # create raw object 
    raw = RawArray(data,info,verbose=False)
    #print(data)
    return raw

def filter_standardization(raw,window_size = 1000,
                          l_freq = 0,h_freq = 30, verbose = False):
    """
    raw: raw object from mnew
    window_size: rolling window_size for standardization,
    l_freq, h_freq: frequency filters
    nClass: the number of event channel to use 
    """

    filtered_X = raw.filter(l_freq=l_freq, h_freq= h_freq, method='fir',phase="minimum",
                            verbose= verbose, picks = [x for x in range(32)])
    filtered_X = filtered_X.to_data_frame().drop(['time'],axis=1)
    # only the first 32 channels to standardize
    filtered_X = filtered_X.iloc[:,0:32]
    filtered_standardized = ((filtered_X - filtered_X.rolling(window_size).mean()) / filtered_X.rolling(window_size).std()).dropna()
    # filtered and stardardized training data
    input_signal = filtered_standardized.to_numpy()
    input_signal = np.swapaxes(input_signal,0,1)

    data = raw.get_data()
    # strip the first 99 data points due to rolling window implementation
    target_signal = target_signal_val =data[32:38,window_size-1:] # export all channels, use only one channel eventually
    #print(input_signal.shape,target_signal.shape)
    
    # reformatt into tensor
#     input_tensor = ttype(input_signal.reshape(1,1,input_signal.shape[0],-1))
#     target_tensor = labeltype(target_signal.reshape(-1))

    #print(input_tensor.shape, target_tensor.shape)
    return (input_signal, target_signal)

def train_model(model, ttype, train_loader, val_loader, 
           optimizer, loss_func, epochs, 
          loss_buffer_size=4, 
        prog_bar=None):
    
    loss_track = {"name":[],
                  "loss":[],
                  "acc":[],
                  "iteration":[],
                  "iteration_time":[]}
    # use this to keep track of progress
    progress_bar = tqdm(range(int(epochs)), bar_format='{l_bar}{bar:5}{r_bar}{bar:-5b}')
    #--------------------- Training --------------------#
    acc = 0
    #iterate through epochs
    niteration = 0
    for e in progress_bar:
        for batch_index, (train_x, labels) in enumerate(train_loader):
            # calucate iteration time
            start = datetime.datetime.now()

            model.train()
            optimizer.zero_grad()

            # make sure the out put is in the right format
            # needs to be in [nbatch x 1 x nfeutures x time]
            # unsqueeze(0) to add back the first dimension if necessary
            ## out = model(sig.unsqueeze(0))

            out = model(train_x)
            
            # permute the out for cross entropy loss
            out = out.permute(0,2,1)
            #print(out.shape, train_ans.shape)
            loss = loss_func(out, labels)

            loss.backward()
            optimizer.step()

            #--- Record name, loss, validation accuracy --#
            loss_track['name'].append(model.__class__.__name__)
            loss_track['loss'].append(loss.mean().detach().cpu().numpy())
            loss_track['iteration'].append(niteration)
            
            # call test_model for validation accuracy calculations
            # ----  update training progress -------------#

            # Update progress_bar
            s = "Epoch: {},Iteration: {}, Loss: {:.8f}, Validation AUC:{} "
            format_list = [e, niteration, loss.mean().detach().cpu().numpy(), acc]         
            s = s.format(*format_list)
            progress_bar.set_description(s)
           
            acc = test_model(model, val_loader)
            # calculate AUC every iteration, can be quite slow
            loss_track['acc'].append(acc)
            niteration += 1
            
            end = datetime.datetime.now()
            iteration_time = end - start
            loss_track['iteration_time'].append(iteration_time)
    return loss_track

def test_model(model, val_loader):
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
        
        try: # incase there is only one class
            auc = roc_auc_score(y_true = y_true,y_score = y_score)

            auc_list.append(auc)
        except:
            auc_list.append(np.nan)
    acc = np.array(auc_list)
        #acc = matthews_corrcoef(y_true = y_true, y_pred = y_pred)
        # return the average
    return np.nanmean(acc)



class EEGDataset(Dataset):
    """
    A pytorch dataset
    input shapes:
        train_x: [nbatch, channels, sequence]
        train_y: [nbatch,  sequence]
    
    Output shape:
        Need to add a magic second dimension in order for Deep_sith
        to work properly
        train_x: [nbatch, 1, channels, sequence]
        train_y: [nbatch,  sequence]
    """
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return self.train_x.shape[0]
    
    def __getitem__(self, idx):
        
        return (self.train_x[idx].unsqueeze(0),
                self.train_y[idx])
    
def split_train_val(train_x_t ,train_y_t,
                    batch_size = 1, train_split = 0.8):
    # batch_size is a hyper parameter to tune 
    dataset = EEGDataset(train_x_t ,train_y_t )

    # get the entire length of the dataset
    dataset_size = len(dataset)

    train_size = int(train_split * dataset_size)
    test_size = dataset_size - train_size

    dataset = EEGDataset(train_x_t, train_y_t)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])



    train_loader = DataLoader(dataset= train_dataset, batch_size=batch_size, 
                             shuffle=False)
    val_loader = DataLoader(dataset= val_dataset, batch_size=batch_size, 
                             shuffle=False)
    return (train_loader,val_loader)
