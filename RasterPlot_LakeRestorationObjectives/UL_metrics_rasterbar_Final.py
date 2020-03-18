# This code reads dataset for a set of lake restoration objectives and plots vesrus water level
# them as a raster plot (heat map)

# This code has been written by Somayeh Sima in 12/1/2019
#----------------------------------------------------
# Import the Pandas and Matplotlib and other packages
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy as sci
import numpy as np
from numpy import *
from scipy.stats import *

metrics_df=pd.read_csv('Objectives.csv', header=0, sep=',',
                       index_col=0, parse_dates=True,
                       infer_datetime_format=True, low_memory=False)

# Determine the range of water level for your data to be plotted
min_level=1270.0
#max_level=max(metrics_df.index)
max_level=1278.0

# determine an increment to unify all metrics
step=1
idx = np.linspace(min_level,max_level,(max_level-min_level)/step+1)

#Bin water level data into classes
bins_space=1
bins_num=(max_level-min_level)/bins_space+1
L_bins=np.linspace(min_level,max_level,bins_num)
#print(L_bins,bins_num)

# Define the name of metrics/objectives
metrics=['EDA','albedo','DistanceBari','DistanceGolmankhaneh','N_S_connection',
         'Island_Connection_to_lands','Isalnad_merging']

#read and merge dataframes of metrics
#create a blank dictionary
metrics_dic={}

#Normalize data
metrics_df_n=(metrics_df-metrics_df.min())/(metrics_df.max()-metrics_df.min())

#Sort data
metrics_df_ns=metrics_df_n

import seaborn as sns;sns.set()

#yticks=idx
#yticks = np.linspace(1270,1278,9)
cbar_val=np.sort(np.linspace(0,1,81))
cbar_val_r=cbar_val[::-1]
sns.set(font_scale=1)
ax = sns.heatmap(metrics_df_ns, cmap='Blues',yticklabels=10,
                 cbar_kws={'ticks':[],'label':'------>\ndesired direction','values':cbar_val_r},
                 linecolor='white')


fig = ax.get_figure()

#ax.set_title("Objectives vs. water level")

# We want to show all ticks...
ax.set_xticks(np.arange(len(metrics_df_ns.columns))+0.45)
#ax.set_yticks( idx )
ax.invert_yaxis()
#ax.set_yticks(bins_num)

#ax.set_xticklabels(metrics_df_ns.columns)
ax.set_ylabel('Lake level (m)')
#ax.set_xlabel('metrics',ha='left')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",va='center',
         rotation_mode="anchor")

#print (ax.get_xlim(),ax.get_ylim())

#Extra x axis: categories of the objectives
ax.set_xlabel('\n                                                                                                                '
              '\n   Human          Water quality                          Ecology      Recreation                     '
              '\n  health                                                                                                                   ',
              ha='center',va='bottom',fontsize=9,weight='bold',color='black')
for _, spine in ax.spines.items():
    spine.set_visible(True)

ax.text(.30,83,' 0        173       0.3     0.15      540       20000      No        No       No        0        0', fontsize=8)
ax.text(.30,-6,'730       373      1.6      0.18       0          0        Yes       Yes       Yes     1150    30000', fontsize=8)


#Plot the border box
ax.axhline(y=0, color='gray',linewidth=3)
ax.axhline(y=81, color='gray',linewidth=3)
ax.axvline(x=0, color='gray',linewidth=3)
ax.axvline(x=11, color='gray',linewidth=3)

# draw vertical line from (70,70) to (100, 250)
plt.plot([1,1], [0,-3],color='gray' ,  linestyle='-', linewidth=1)


plt.show()
#fig.tight_layout()
fig.savefig('Objectives_RasterPlot_Blue.png')


