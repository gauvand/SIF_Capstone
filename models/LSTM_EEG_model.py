# pytorch
import torch
import torch.nn
import torch.nn.functional as F
from torch import nn as nn


class LSTM_EEG(torch.nn.Module):

    def __init__(self, in_features, hidden_dim, out_feuture, num_layers, dropout):
        super(LSTM_EEG, self).__init__()
        
        self.hidden_dim = hidden_dim

        self.lstm = torch.nn.LSTM( in_features,hidden_dim,num_layers =num_layers, 
                                  batch_first=True, dropout = dropout)

        # The linear layer that maps from hidden state space to output space
        self.fc = torch.nn.Linear(hidden_dim, out_feuture)

    def forward(self, inp):
        # input of shape consistent with Deep+isith
        # input shape: (batch_size, 1,inputFeuture_size, nSequence)
        
        # output shape: (batch_size, )
        inp_reshape = inp.squeeze(1).permute(0,2,1)
        # should be reshaped to (batch_size, nSequence, inputFeuture_size) like [1, 6000, 32]
        #print(inp_reshape.shape)
        lstm_out, _ = self.lstm(inp_reshape)
        #print(lstm_out.shape)

        out_space = self.fc(lstm_out)
        #print(out_space.shape)

        return out_space