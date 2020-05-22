"""
Plotting ERPs is useful in neuroscience. This is an event-related potential and is an average of a signal over n trials
These can be easily computed using EEGLAB
EEGLAB is a GUI - making accessing the raw data slightly tricky, but the following MATLAB script can help.

Following the MATLAB code, python code, using the seaborn package. This package is good for plotting

The graphs produced in MATLAB aren't easily changed for publication purposes, hence the replotting
"""

# Matlab code:
# Use this script to access the data from the .fig.
# The figure needs to be open in Matlab, for the script to run properly. The data can be exported to an excel file
# saved as .csv for input to the plotting_ERP function

# h = findobj(gca, 'Type', 'line')
# x = get(h, 'Xdata');
# y = get(h, 'Ydata');

# Plotting with seaborn in python
import sys

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


erp_data = r'\ERPs.csv'
erp_df = pd.read_csv(erp_data)

# Set up plot for "Left ERP"
# The x and y labels correspond to the headings in the pandas dataframe
erp_left = sns.lineplot(x='left_x',
                        y='left_y',
                        data=erp_df,
                        # color='b'
                        ).set_title('Cz Left')

# Set up plot for "Right ERP"
# erp_right = sns.lineplot(x='right_x',
#                         y='right_y',
#                         data=erp_df,
#                         # color='b'
#                         ).set_title('Cz Right')
# x and y labels.
plt.xlabel('Time(ms)')
plt.ylabel('Potential(\u03BCV)') # \u03BC Unicode for 'micro'

# "M"
plt.axvline(x=0, color='k')
# "W"
plt.axvline(x= -108, color='k', linestyle='--' )

plt.axhline(y=0, color='k')
# Correspond plot with original Libet paradigm
plt.gca().invert_yaxis()
plt.ylim(2.5, -2.5)
plt.xlim(-1500, 500)
plt.show()


