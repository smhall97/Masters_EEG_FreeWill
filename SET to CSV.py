"""
Author: Siobhan Hall (edited by Ben Wolfaardt)
Based off the Libet experiment code by Mikkel C. Vinding
"""

import mne
import pandas as pd
import numpy as np
import csv
import os

#Each Libet clock cycle = 1 epoch (size of epoch to be determined)
# Each cycle is variable length - might need to use FieldTrip tool box to extract variable lengths, or use average trial time plue one/two SD of the mean?

#101 = left, 102 = right
Triggers = [101, 102]

#Epoch file naming convention: 
# Letters in alphabetical order of collection - initial record is with a number code, so no correlation to person's name. 
# Names = ['A']
# Names = [ 'C', 'E']
# Names = ['A', 'B']
Names = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U'] 

#Directory where the file should be saved
out = r"C:\\Users\\Siobhan\\Desktop\\SET\\CSV\\"
#out = os.path.join(r"C:\Users\Siobhan\Desktop\SET",r"CSV")
for name in Names:
    for trigger in Triggers:
        try:
            df = mne.io.read_epochs_eeglab('D:\\MATLAB\\eeglab14_1_2b\\data\\000_twoseconds\\{}_{}.set'.format((name),(trigger)))
            # df = mne.io.read_epochs_eeglab('C:\\Users\\Siobhan\\Desktop\\SET\\{}_{}.set'.format((name),(trigger)))

            i = 0
            for epoch in df:
                i=i+1
                dfNew1 =  pd.DataFrame(epoch)
                newstring = out + str(name) + "_" + str(trigger) + "_" + format(i, '03d') + ".csv"
                np.savetxt(newstring,dfNew1,delimiter=',')
                #np.savetxt(out + str(name) + "_" + str(trigger) + "_" + format(i, '03d') + ".csv",dfNew1,delimiter=',')
                #print(newstring)
        except OSError:
            pass
        # except IOError == False:
        #     print("Error")

    print("Triggers completed for participant")
print("Participant completed")
                          
