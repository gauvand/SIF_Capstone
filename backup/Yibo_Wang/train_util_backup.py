# not using dataloader
def train(model, ttype, train_sig, train_ans, 
          test_sig, test_ans, optimizer, loss_func, 
          epoch, loss_buffer_size=4, batch_size=1, 
        prog_bar=None):
    """
    **Training**:
    Currently only use one batch. Will add batch processing ability later.     
    **Validation**:   
    The performance metric used here is **matthews correlation coeefficient** since the data are unbanlanced.  
    Signals need to be in correct format. validation input: [nbatch x 1 x nFeutures x time] tensor.
    The validation target has dimension of [time] tensor, in which each entry should be binary at any time point.  
    
    """
    assert(loss_buffer_size%batch_size==0)
    
    loss_track = {"name":[],
                  "loss":[],
                  "acc":[],
                  }
    # make sure tensor types are correct
    #train_sig = train_sig.type(torch.DoubleTensor)
    #train_ans = train_ans.type(torch.LongTensor)
    
    #--------------------- Training --------------------#
    #currently we only have 1 batch, so no batch processing.
    model.train()
    
    #print(target.shape)
    optimizer.zero_grad()
    
    # make sure the out put is in the right format
    # needs to be in [nbatch x 1 x nfeutures x time]
    # unsqueeze(0) to add back the first dimension if necessary
    ## out = model(sig.unsqueeze(0))
    out = model(train_sig)
    # permute out to the correct format
    out = out[-1,:,:]
    #print(train_ans.shape)
    #print(out.shape)
    loss = loss_func(out, train_ans)
    #print(loss)
    loss.backward()
    optimizer.step()

    
    #--- Record name, loss, validation accuracy --#
    loss_track['name'].append(model.__class__.__name__)
    loss_track['loss'].append(loss.mean().detach().cpu().numpy())
    
    # call test_model for validation accuracy calculations

    acc = test_model(model, test_sig,test_ans)
    loss_track['acc'].append(acc)
       
    # ----  update training progress -------------#
    if  prog_bar is not None:
        # Update progress_bar
        s = "Epoch {},Loss: {:.8f}, Validation AUC{}: "
        format_list = [epoch, loss.mean().detach().cpu().numpy(), acc]         
        s = s.format(*format_list)
        prog_bar.set_description(s)

    return loss_track

def test_model(model, signal,target):
    """
    Test for accuracy
    Iterate through each batch and make prediciton and calculate performance metrics
    Use **matthews correlation coeefficient** since the data are imbanlanced
    Again 
    Signals need to be in correct format. validation input: [nbatch x 1 x nFeutures x time] tensor.

    The target has dimension of [time] tensor, in which each entry should be one of the numbers in 
    {0,1,2, ... K} at any time point.  
    
    """
    matthew =[]
    
    #signal = signal.type(torch.DoubleTensor)
    #target = target.type(torch.LongTensor)
    #print(targets.shape)
    out = model(signal)
    # permute out to the correct format
    out = out[-1,:,:]
    # pass through a softmax to tansform to probability
    res = torch.nn.functional.softmax(out, dim=1)
    #print(res.shape)
    # The class with 1 as label
    y_pred = res[:,1].detach().cpu().numpy()
    y_true = target.detach().cpu().numpy()
    auc = roc_auc_score(y_true = y_true, y_score = y_pred)

    return auc