import csv
import numpy as np
import pandas as pd

person = '22'
Trials = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10','11', '12']
# ,'11', '12', '13'
def combine_csvs(experiment):
    data = {}
    W_time_list = []
    reaction_time_list = []
    with open(f'subject_{person}{experiment}.csv') as f:
                    
        reader = csv.reader(f, delimiter=';', quotechar='"') #Import csv file

        trial = 2
        
        for row in reader:
            firstline = True
            for row in reader: 
                if firstline == True:
                    firstline = False 
                else:
                    if row:
                        pressAngle = float(row[7])
                        pressTime = float(row[6])
                        W_angle = float(row[8])
                        userError = row[-1] 

                        if userError == 'False':  
                            
                            if pressAngle < 90 and W_angle > 360: #this is an occasional "glitch" that might occur if the dot is around '12o'clock' when the person clicks
                                W_angle = W_angle - 360
                                reaction_angle = pressAngle - W_angle
                           
                            else: 
                                reaction_angle = pressAngle - W_angle #this to have an average point of W. The reaction time is the time ...
                                                                #between the reported moment of conscious awareness and the press. 
                            
                            #reaction time - time between W and M - need to find the same dictionary keys and then from there to find the exact time stamp (i.e. when the dot started )
                            reaction_time = reaction_angle*(2560/360)
                            reaction_time = reaction_time/1000 
                            #reaction_time needs to be averaged out over all the successful trials. This is basically the same as W_time, ...
                            #but W_time is fed individually into a separate dictionary. 
                            
                            W_time = pressTime - reaction_time #for the dictionary. Only in the else case is it not included.
                            
                        else: #when userError is True
                            pressTime = 0
                            
                            W_time = 0 #user error and thus this trial should be discarded 
                    
                        
                        data[trial] = {
                            'pressTime': pressTime,
                            'W_time': W_time,
                            'reaction_time': reaction_time,
                        }

                        trial += 1

                        # if W_time != 0:
                        #     W_time_list.append(W_time)
                        if reaction_time > 0:
                            reaction_time_list.append(reaction_time)
    print(np.mean(reaction_time_list))
    with open(f'subject_{person}{experiment}_triggers.csv') as f:
        reader = csv.reader(f, delimiter=';', quotechar='"')

        firstline = True
        firstline_trigger_time = 0

        for row in reader:
            trigger_time = float(row[0]) #this time correlates to the time indexed at 0 in the eyegaze csv's
            label = row[1] #string
            
            trial = int(row[2]) #same type as other "counts"?

            if label == 's': #not to be included, this is the time of the start of the Libet clock 
                if firstline:
                    firstline_trigger_time = trigger_time
                continue
            #Participant chooses left or right    
            if label == 'l' or label == 'r':    

                if trial in data: 
                    data[trial]['label'] = label
                    data[trial]['trigger_time'] = trigger_time
                else:
                    data[trial] = {
                        'label' : label,
                        'trigger_time' : trigger_time,
                    }
                
                if firstline:
                    firstline = False
                    data[trial]['pressTime'] = trigger_time - firstline_trigger_time
                    data[trial]['W_time'] = 0
                    data[trial]['reaction_time'] = np.mean(reaction_time_list)
                    
                    
    return data

all_data = []
for t in Trials:
    d = combine_csvs(t)

    sorted_keys = sorted(list(d.keys()))
    for k in sorted_keys:
        all_data.append(d[k])

# for r in all_data:
    # print(r)
    #print(type(r))

df = pd.DataFrame(all_data)
# print(df)
pt_info = df.to_csv (r'C:\Users\Siobhan\Desktop\SET\Libet_information\subject_22_Libet_information.csv.', index = None, header=True)

#all_data = dictionary to be read to a csv
# with open(f'subject_{person}_Libet_information.csv', 'wb') as f:
#     w = csv.DictWriter(f, r.keys())
#     w.writeheader()
#     w.writerow(r.value())