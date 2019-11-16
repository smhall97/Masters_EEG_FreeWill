# Masters_EEG_ET

Libet_information: 
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

Median_filter
- This is a pre-processing step for the gaze data
- The filter will be applied to the data in a numpy array format. 

Fixation Identification Algorithm 
- This will be the code to iterate over the eye tracking data looking for fixations within each cycle.
- This is based on the assumption that there are correlations between the eyes (fixations, pupil changes, gaze etc.) that 
  that correspond to changes in cognition / attention.
- These data are collected using the Tobii Pro SDK. Documentation (Python 2.7 and Python  3.5; Matlab 2016A & B, 2015B; C; 
  Unity on Windows specifically regarding this can be found on their website: https://developer.tobii.com/tobii-pro-sdk/


Separate_gaze:
- This is the pre-processing stage for the fixation identification algorithm described above. 
Separate_pupil:
- Pupil dilation can be considered too noisy for reliable results, so at this point I am unsure if this will even be used in the overall project. 

SET to CSV
- All eeglab files are saved as .set files.
- There are two steps to get the data file ready for input into the model:
a) .set to .csv (each row in the csv corresponds to a channel, and each column will relate to one time point [making up the length of the epoch])
b) From the .csv file to a .pickle format, but this will occur in a separate script, i.e. "CSV to pickle". 

csv_pickle
- This is the last data preparation phase before inputting the data into the model for training (in "DeepLearning_EEG")
- It is important to ensure these files are set up correctly. 
-This file will be deleted - as this is only the pseudocode. The exectuable code is saved under "new_pickling_v2"

new_pickling_v2
- This takes the csv files for various stages of data preparation. Data preparation is the stage before the models.
- The stages are outlined as follows:
    a) Separated based on person (code) and decision ("left" / "right")
    b) Separate each window / epoch into frames: frames are based on % values to account for variable window sizes
    c) There are two ways to send the data through the model. These are as follows:
        i] Separated = separated on a person level = train on some participants; test test on others
        ii] Grouped = mixed between persons; separation on trial level: train on some, test on some 

DeepLearning_EEG
- This is very much in the initial stages. 
- The intitial model will be a convolution neural network.
- The model's task is to classify the action (L/R) at each time window preceding the action using the EEG data. 



