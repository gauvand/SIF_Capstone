"""
General-purpose training script for Grasp-and-lift EEG data prediction and classification
Currently support:
1. A base model for LSTM (Rivanna GPU)
2. Deep_isith module
**Use Kernal PyTorch 1.4.0 Py3.7**  
parts are from **Neural Network Example**(Authors: Brandon G. Jacques and Per B. Sederberg)

Yibo Wang

"""

from src.train_util import *
from models.Deep_isith_EEG_model import *
from models.LSTM_EEG_model import *
import pandas as pd
# read config file
import configparser
import argparse

# enable use of command line
parser = argparse.ArgumentParser(description='Input config files')
parser.add_argument('--config', default = 'config/training_config.ini', type=str,
                    help='an integer for the accumulator')
opt, _ = parser.parse_known_args()

# parser to read parameters
config = configparser.ConfigParser()
config.sections()

# parameters from config file
config.read(opt.config)
dir = config['data']['directory']
subject_num = int(config['data']['subject #'])
kernel_size = int(config['training']['kernel_size'])# sliding window size to use
step = int(config['training']['step']) #  --the step between each slice. means overlap between batches is 1- step 
modelName = config['training']['model']
# num of epochs to train
nepochs = int(config['training']['nepochs'])
loss_func =  torch.nn.CrossEntropyLoss()
batch_size = int(config['training']['batch_size']) # batch_size is a hyper parameter to tune 
train_split = float(config['training']['train_split'])
lr = float(config['training']['lr'])


for j in range(1,13): # out loop train all subjects
    results = []
    subject_num = j
    # load data and do preprocessing
    train_x_list = []
    train_y_list = []
    print(f"Sarting to load Subject{subject_num} Data.")
    for file in os.listdir(dir):
        sub_idx = file.find('_')
        if file[:-4].endswith('_data') & (file[4:sub_idx] == str(subject_num)):
            raw = creat_mne_raw_object(dir+file,read_events=True)
            # filter all channels
            input_signal,target_signal = filter_standardization(raw,window_size = 1000,
                                l_freq = 0,h_freq = 30)

            input_tensor = ttype(input_signal.reshape(1,1,input_signal.shape[0],-1))
            target_tensor = labeltype(target_signal.reshape(6,-1)) # should be six channels
            input_tensor = input_tensor.squeeze()
            # patches data 
            patches_train = input_tensor.unfold(dimension = 1, size = kernel_size, step = step).permute(1,0,2)
            patches_label = target_tensor.unfold(1, kernel_size, step).permute(1,0,2)
            #print(patches_train.shape, patches_label.shape)

            # append to a list
            train_x_list.append(patches_train)
            train_y_list.append(patches_label)  
    print("Finished! {} data are loaded and preprocessed".format(len(train_x_list)))

    # concatenate them
    train_x_t = torch.cat(train_x_list, dim=0)
    train_y_t = torch.cat(train_y_list, dim=0)
    print(train_x_t.shape, train_y_t.shape)

    # start training, iterate thorugh events
    for i in range(1,7): # There are six events 1 - 6
        nClass = i - 1
        train_y_t_nClass = train_y_t[:,nClass,:]
        # create dataloader class
        train_loader,val_loader = split_train_val(train_x_t ,train_y_t_nClass,
                        batch_size = batch_size, train_split = train_split)

        # match with modelsm currently model name has to be exact
        if modelName == 'Deep_isith':
            #--------------- sith layer model parameters ------------------#
            # make sure this in_features matches the number of feutures in the EEG data
            sith_params1 = {"in_features":32, 
                            "tau_min":1, "tau_max":150, 
                            "k":15, 'dt':1,
                            "ntau":8, 'g':0.0,  
                            "ttype":ttype, 
                            "hidden_size":25, "act_func":nn.ReLU()}

            sith_params2 = {"in_features":sith_params1['hidden_size'], 
                            "tau_min":1, "tau_max":150.0, 'buff_max':600, 
                            "k":4, 'dt':1,
                            "ntau":8, 'g':0.0, 
                            "ttype":ttype, 
                            "hidden_size":25, "act_func":nn.ReLU()}
            layer_params = [sith_params1, sith_params2]

            #------------------ model configuration ------------------------#
            # number of output feature should be 2 since we always train one at a time, so now 1+1
            model = DeepSITH_Tracker(out=2,
                                        layer_params=layer_params, 
                                        dropout=0.1).double()
        elif modelName == 'LSTM':
            hidden_size = 50 # try 256  later
            # make sure this in_features matches the number of feutures in the EEG data
            model = LSTM_EEG(in_features = 32, hidden_dim = hidden_size, 
                              out_feuture = 2,num_layers =3, dropout=0.1).double()
        else:
            print('Model name not recognized!')

        optimizer = torch.optim.Adam(model.parameters())
        # map model to GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        #------------------- start training ---------------------------#
        perf = []
        perf = train_model(model, ttype, train_loader, val_loader,
                        optimizer, loss_func, epochs=nepochs)
        results.append(perf)


        if not os.path.exists('models'):
            os.makedirs('models')
        PATH = f'./models/{modelName}_Subject{str(subject_num)}_numEvent{nClass}.pth'
        torch.save(model.state_dict(), PATH)


    # save results
    df = pd.DataFrame()
    event = ['HandStart','FirstDigitTouch','BothStartLoadPhase',
                'LiftOff','Replace','BothReleased']
    for i in range(len(results)):
        perf = results[i]
        new_df = pd.DataFrame(perf)
        new_df['event'] = event[i]
        df = df.append(new_df)
    if not os.path.exists('csv'):
        os.makedirs('csv')
    csv_name = f'./csv/{modelName}_Subject{str(subject_num)}.csv'
    df.to_csv(csv_name)
