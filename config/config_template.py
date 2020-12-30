"""
A template to write training config
Currently support:
1. A base model for LSTM (Rivanna GPU)
2. Deep_isith module
"""

import configparser
config = configparser.ConfigParser()
config['data'] = {"directory":'./grasp-and-lift-eeg-detection/',
                  "subject #": "1"
    
}

config['training'] = {'model': 'Deep_isith', # --[LSTM, Deep_isith]
                     'kernel_size': '50000', # --sliding window size to use
                      'step' : '45000', #  --the step between each slice. means overlap between batches is 1- step 
                    
                    'nepochs' : '20' # -- num of epochs to train
                      }

with open('training_config.ini', 'w') as configfile:
    config.write(configfile)