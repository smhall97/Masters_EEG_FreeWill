# Masters_EEG_ET

## Libet_information: 
- This is where the file will have everything running with regards to the triggers and the EEG times
- Triggers refers to the "M" decision, i.e. did the person choose left or right.
- W_time is the self-reported moment of conscious awareness using the Libet clock 
- M_time is the time of the action (i.e. exact time stamp of the trigger).
- Each participant has 11 sets of 11 trials (trials can be variable on account of the PsychoPy running out of memory).
- A trial refers to one cycle of the Libet clock
- A set refers to consecutive cycles of the Libet clock recorded in a single csv. 
- The experiment refers to the collection of 11 or 12 sets, which is made up of 11 csvs.
- It is possible for a set to have <11 trials, as if the person took long to decide, the programme would run of memory.
- Collectively, each participant would have 110 trials at least. 

- The code for the Libet experiment can be found in the following repository, by Mikkel C. Vinding: https://github.com/mcvinding/clock_test


## SET to CSV
- All eeglab files are saved as .set files.
- There are two steps to get the data file ready for input into the model:
a) .set to .csv (each row in the csv corresponds to a channel, and each column will relate to one time point [making up the length of the epoch])
b) From the .csv file to a .pickle format, but this will occur in a separate script, i.e. "CSV to pickle". 


## CNN_model_final 
- As per DL convention, the data are split into a training, validation and test sets. 
- This code splits the data into the three subsets based on two methods:
    a) Separated: some participants are placed in the training set, other data in the validation set and the remainign participants are placed in the test sets. 
    b) All the data are pooled before being randomly split into the three subsets. 
- The model's task is to classify the action (L/R) at each time window preceding the action using the subconscious EEG data. 
- The subconscious EEG data can be obtained by creating frames of 25 %, 50 %, 75 % of the epoch that will be fed on the model. 
- The DL model is used is based on the Convolutional Neural Network (CNN) created by Schirrmeister et al (2017) - Deep Learning with Convolutional Neural Networks 
for EEG Decoding and Visualisation


## plotting ERP
- replotting the readiness potential from the EEGLAB figure

## eyetrackingscript
- Accessing the different csvs collected during data collection to access trial times and trigger information
