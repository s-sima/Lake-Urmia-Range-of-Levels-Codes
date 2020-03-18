# This code reads a results of WQ samplings from a lake at various time and locations by
# several researchers
# The code does the following steps#
# 1) reads the .csv files
# 2) calculates the mean and standard deviations of samples at taken a particular date (by all researcher)
# 3) fills the gap between dates to have a consistent data set from the beginning to the end of all samplings
# 4) plots a time series of the variables versus water level

# This code has been written by Somayeh Sima in 6/1/2019
#----------------------------------------------------
# Import the Pandas and Matplotlib and other packages
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sci
import numpy as np
from numpy import *
from scipy.stats import *
import seaborn as sns

# Determine the temporal range of your data
beginDate = '06-01-1977'
endDate = '25-08-2017'
idx = pd.date_range(beginDate, endDate)


# Define the name of authors who you are using their data as a unique list
r=['Kelts and Shahrabi,1986', 'Daneshvar& Ashasi,1995',  'Alipur,2006',
    'Hafezieh, 2016','Asem et al.,2007','Karbasi et al.,2010', 'sima & Tajrishy, 2015', 'EAWB']

# Define the name of water quality parameters (header line of the .csv files)
Parameters_list=['water_level','TDS','Na','Mg','SO4','Cl','HCO3','K','Ca']

#read and merge dataframes of ionic composition and TDs
dicMean={}
dicStd={}
df={}

#create a blank dictionary for each parameter
for n, p in enumerate(Parameters_list):
    globals()['Mean_%s'%p]={}
    globals()['Std_%s'%p]={}

# Notice that the name of csv files is the first 6 letters of the reference (authors)
for i , dr in enumerate(r):
        dfname=(dr[0:6])
        # set the path of your .csv files
        csv_FilesPath = r'C:\Users\somay\PycharmProjects\PyCodes\ULplots_PNAS\TDS_Ions_Timeseries\WQCSV_InputFiles\\'
        name = dfname.strip() + '.csv'
        df = pd.read_csv(csv_FilesPath + name, header=0, sep=',',
                         index_col=0, parse_dates=True,
                         infer_datetime_format=True, low_memory=False)

        grouped_df = df.groupby('date')
        df_mean = df.groupby(df.index.date).mean()
        df_std = df.groupby(df.index.date).std()

        dicMean[dfname] = df_mean.reindex(idx)
        dicStd[dfname] = df_std.reindex(idx)

# Select desired columns and form a dictionary for each parameter
        for n, p in enumerate(Parameters_list):
            globals()['Mean_%s'%p][dr]=dicMean[dfname][p]
            globals()['Std_%s'% p][dr] = dicStd[dfname][p]

# convert the dictionary of each parameter to a data frame named for example df_MeanTDS
#print(Meanwater_level)
for n, p in enumerate(Parameters_list):
      # Selecting desired parameters (columns)
    globals()['df_Mean%s'%p]=pd.DataFrame.from_dict(globals()['Mean_%s'%p])
    globals()['df_Std%s' % p] = pd.DataFrame.from_dict(globals()['Std_%s' % p])

##Arbituary print of the TDS statistics
#print(df_MeanTDS.min(),df_MeanTDS.max(),df_MeanTDS.mean(),df_MeanTDS.std())

# Put together all TDS & Volume data as time series regardless of the authors
df_MeanTDS_allAuthours=df_MeanTDS.stack()
df_Meanlevel_allAuthours=df_Meanwater_level.stack()

# concat two series of Level and TDS and sets the columns name of the resulting dataframe
L_TDS_Timeseries=pd.concat([df_Meanlevel_allAuthours.rename('water level'),
                           df_MeanTDS_allAuthours.rename('TDS')], axis=1)

#Determine the x and y dataset
#x_data=water_level , y_data=TDS

x_data=array(df_Meanlevel_allAuthours)
y_data=array(df_MeanTDS_allAuthours)

#mask NaN data from x and y datasets
mask = ~np.isnan(x_data)&~np.isnan(y_data)

LTDS_df=pd.DataFrame({"level":x_data[mask],"TDS":y_data[mask]})

# sort data by water level values
LTDS_df.sort_values(by=['level'])

# bining x axis (water level) values
min_l=1270.0
max_l=1278.5
tickspace=0.5
num=(max_l-min_l)/tickspace+1
L_bins=np.linspace(min_l, max_l,num)


#Classify df based on bins
LTDS_df['binned'] = pd.cut(LTDS_df['level'], L_bins)

# Plot TDS/Level box plot
fig = plt.figure(figsize=(4,3.2), dpi=900)
ax = fig.add_subplot(111)
# Plot TDS/Level box plot
#sns.set(font_scale =1)
flierprops=dict(marker='o', markersize=2,markerfacecolor='deepskyblue',markeredgewidth=1,markeredgecolor="black")

sns.boxplot(x="binned",y="TDS",data=LTDS_df,color='deepskyblue',width=0.6,
            linewidth=1.2,flierprops=flierprops)

#tickmark
#put tick marks from the lower band to upper band with desired increments
ylabel=list(range(100, 440, 40))
plt.yticks(ylabel)

# Set the limits of the axes
#xlim
locs, labels = plt.xticks()
labels =L_bins

plt.xticks(locs, labels)
ax.set_xticklabels(labels, rotation=50,fontsize=8,fontname="Arial",color="black",weight="bold")
plt.ylim(100,440)

ax.set_yticklabels(ylabel,fontsize=8,fontname="Arial",weight="bold")
#ax.tick_params(axis='x', pad=0.2)

# label the axis
plt.ylabel('TDS (g L$^{-1}$)',fontsize=9,fontname="Arial",weight="bold")
plt.xlabel('Water level(m)',fontsize=9,fontname="Arial",weight="bold")

#ax.grid(True)

# write an arbitrary text on the plot
plt.figtext(0.58,0.9,"--- Ecological target level",fontsize=8,fontname="Arial",weight="bold", color="red")

# draw an arbitrary line from (x1, x2) to (y1, y2)
plt.plot([8.1,8.1], [0, 440], color='red', linestyle='--', linewidth=1)


plt.show()
fig.savefig('TDS_L_BoxWhisker.png')
