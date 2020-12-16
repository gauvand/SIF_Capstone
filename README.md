## MSDS Capstone: Scale Invariant Factors for Brain-Computer Interfaces 
### University of Virginia | School of Data Science 

This repository tracks the use and validation of a specialized LSTM network (Scale Invariant Temporal History (SITH)) that models the scale-invariant patterns of the brain's time cells. SITH's development was pioneered by the Computational Memory Lab at the University of Virginia; more information and code can be found here: https://github.com/compmem/SITH_Layer. Our goal is to quanitfy how well SITH can identify actions given EEG data in hopes to improve prediction accuracy of current BCI (Brain-Computer Interface) technology on EEG data by implementing neural features such as time cells as artifical agents. 

All work in this repository belongs to Gaurav Anand (ga7er@virginia.edu), Yibo Wang (yw9et@virginia.edu), Arshiya Ansari (aa9yk@virginia.edu), and Beverly Dobrenz (bgd5de@gmail.com). 

1. Preliminary Result (12/15/2020)

	**Methods**  
	For now, only consider one subject (subject1) for modeling. Predict only one event/channel a time (since there are events overlapping), and incorporate sliding-window standardization and filtering.
	Load all eight events and split into 80% training and 20% validation/holdout set.
	Use dataset and dataloader pytorch classes to control batch processing.
	Use sequence length of 50000 and stride size of 5000 (overlaped sequence for each batch)for each minibatch.

	**Note**  
	The code is tested on Rivanna with GPU. (may needs some tweaks with CPU only)
	Need train_util.py and Deep_isith_EEG.py helper functions as well as the SITH_Layer_master package
	
	![plot](./subject1_result_deep_isith.png)
	
	**Result**
	Average AUC on the holdout set for all six events for Subject1 is **0.97**
