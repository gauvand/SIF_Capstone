"""
### A base model for LSTM (Rivanna GPU)
**Use Kernal PyTorch 1.4.0 Py3.7**  
parts are from **Neural Network Example**
(Authors: Brandon G. Jacques and Per B. Sederberg)

Yibo Wang

"""

from train_util import *
from Deep_isith_EEG import *
from base_models import *
import pandas as pd

# parameters
results = []
dir = "./grasp-and-lift-eeg-detection/"
kernel_size = 50000 # sliding window size to use
step = 45000 #  --the step between each slice. means overlap between batches is 1- step 
# num of epochs to train
nepochs = 20
# Just for visualizing average loss through time. 
loss_buffer_size = 100
loss_func =  torch.nn.CrossEntropyLoss()
batch_size = 2 # batch_size is a hyper parameter to tune 
train_split = 0.8
lr = 0.01


# start training, iterate thorugh events
for i in range(1,7): # There are six events 1 - 6
    nClass = i

    train_x_list = []
    train_y_list = []

    for file in os.listdir(dir):
        if file[:-4].endswith('_data'):
            raw = creat_mne_raw_object(dir+file,read_events=True)
            input_tensor,target_tensor = filter_standardization(raw,window_size = 1000,
                              l_freq = 0,h_freq = 30,nClass = nClass)
            input_tensor = input_tensor.squeeze()
            # patches data 
            patches_train = input_tensor.unfold(dimension = 1, size = kernel_size, step = step).permute(1,0,2)
            patches_label = target_tensor.unfold(0, kernel_size, step)
            #print(patches_train.shape, patches_label.shape)

            # append to a list
            train_x_list.append(patches_train)
            train_y_list.append(patches_label)  
    print("Finished ! {} data are loaded".format(len(train_x_list)))
    print("Processing event number {}".format(nClass))
    
    # concatenate them
    train_x_t = torch.cat(train_x_list, dim=0)
    train_y_t = torch.cat(train_y_list, dim=0)
    print(train_x_t.shape, train_y_t.shape)


    # create dataloader class
    train_loader,val_loader = split_train_val(train_x_t ,train_y_t,
                    batch_size = batch_size, train_split = train_split)
    
    

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

    optimizer = torch.optim.Adam(model.parameters())
    # map model to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    #------------------- start training ---------------------------#
    perf = []
    perf = train_model(model, ttype, train_loader, val_loader,
                    optimizer, loss_func, epochs=nepochs)
    results.append(perf)

    
    
# save results
df = pd.DataFrame()
event = ['HandStart','FirstDigitTouch','BothStartLoadPhase',
            'LiftOff','Replace','BothReleased']
for i in range(len(results)):
    perf = results[i]
    new_df = pd.DataFrame(perf)
    new_df['event'] = event[i]
    df = df.append(new_df)

df.to_csv('deep_isith_training_output.csv')