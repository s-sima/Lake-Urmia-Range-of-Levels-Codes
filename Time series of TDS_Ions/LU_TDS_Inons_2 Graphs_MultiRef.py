# This code reads water quality data of Lake Urmia, Iran reported by a number of researchers
# (at various time and locations)
# This code does the following steps#
# 1) read the .csv data files
# 2) calculate the mean and standard deviations of samples at taken a particular date (by all researcher)
# 3) fills the gap between dates to have a consistent data set from the beginning to the end of all samplings
# 4) plots a time series of the water quality variables so that the researcher and parameters can be
# distinguished by different markers and colors, respectively.

# This code has been written by Somayeh Sima in 1/28/2019
#----------------------------------------------------
# Import the Pandas and matplotlib packages

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# define the start and the end of time series
beginDate = '06-01-1976'
endDate = '25-08-2018'
idx = pd.date_range(beginDate, endDate)

# show the name of authors as a unique list
r=['Kelts and Shahrabi,1986', 'Daneshvar& Ashasi,1995', 'Ghaheri et al.,1999', 'Alipur,2006'
    ,'Karbasi et al.,2010', 'Sima & Tajrishy, 2015', 'EAWB','Hafezieh, 2016']

#define name of water quality parameters in a list
Parameters_list=['TDS','Na','Mg','SO4','Cl','HCO3','K','Ca']

# Define a list of a symbol for each authors in the r_ions list
marker_list=['v','*','d','^','+','p','x','.']

# Define a list of a color for each parameter
color=['black','r','b','g','brown','purple','orange','gray']

#read and merge dataframes of ionic composition and TDS
dicMean = {}
dicStd = {}
df = {}

# create a blank dictionary for each parameter
for n, p in enumerate(Parameters_list):
    globals()['Mean_%s' % p] = {}
    globals()['Std_%s' % p] = {}

# Notice that the name of csv files is the first 6 letters of the reference (authors)
for i, dr in enumerate(r):
    dfname = (dr[0:6])
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

    # Select desired columns from dictionary for each parameter
    for n, p in enumerate(Parameters_list):
        globals()['Mean_%s' % p][dr] = dicMean[dfname][p]
        globals()['Std_%s' % p][dr] = dicStd[dfname][p]

    # convert the dictionary of each parameter to a dataframe named for example df_MeanTDS
for n, p in enumerate(Parameters_list):
    # Selecting desired parameters (columns)
    globals()['df_Mean%s' % p] = pd.DataFrame.from_dict(globals()['Mean_%s' % p])
    globals()['df_Std%s' % p] = pd.DataFrame.from_dict(globals()['Std_%s' % p])

# Read the CSV file of level into a Pandas data frame object
df_level = pd.read_csv(csv_FilesPath +'Level.csv', header=0, sep=',',
                 index_col=0, parse_dates=True,infer_datetime_format=True, low_memory=False)

# Get a subset of data in a new data frame to visualize
# you can select a specific range of your dataframe for plots by changing the beginDate and endDate
df_level_sub=df_level[beginDate:endDate]

# Save your dataframe of each parameter(which organized based on data from different researchers ) as csv file
#df_MeanTDS.to_csv('AllTDS_DataSet.csv')

# Plot TDS/Level time series
# set the figure size and resolution
fig = plt.figure(figsize=(4.,3.9),dpi=1000)

# Get the current axis of the plot and
# set the x and y-axis labels
#ax1 & ax2 are for the fisrt pannel (TDS-L) and ax is for ions pannel
# Add the first subplot to the figure
ax1 = fig.add_subplot(211)
ax2 = ax1.twinx()

df_level_sub.plot(y='level', kind='line', use_index=True,markersize=0, style='-',sharex=True,
          ylim=[1269, 1279],fontsize=7, marker='.',ax=ax2,legend=True, label="Lake level")
handels,legend4=ax2.get_legend_handles_labels()
legend_properties = {'size':7,'weight':'regular'}
ax2.legend(loc='upper left',frameon=False,prop=legend_properties)

plt.minorticks_off()
# Get the current axis of the plot and
# set the x and y-axis labels

ax1.set_ylabel('TDS (g L$^{-1}$)', fontsize=8,fontname="Arial",weight="bold")
ax2.set_ylabel('Lake level(m)',fontsize=8,fontname="Arial",weight="bold" )

#TDS Plot can be distingusished based on the authours listed in r
#Legend detrmining authors
for m, n in enumerate(r):
   df_MeanTDS.plot(y=n,kind='line', use_index=True, linestyle='None', markersize=2, ax=ax1,ylim=[100, 450], color='brown',
                                   marker=marker_list[m], markerfacecolor='none',markeredgewidth=0.6,fontsize=7,legend=False)

# Add a legend with some customizations
TDSmarker=mlines.Line2D([], [], linestyle='None',color='brown', marker='.',markersize=3, markerfacecolor='none',label='TDS')
#legend2 = ax2.legend(loc='upper left',fontsize='small',facecolor='w', frameon=False,shadow=False,title_fontsize='large')
legend1 = ax1.legend(handles=[TDSmarker],bbox_to_anchor=(0.25, 0.88),frameon=False, shadow=False,prop=legend_properties)

#Plot ions
x={}
for a,b in enumerate(Parameters_list[1:5]):
# Divide the figure into a 2x1 grid, and give me the first section
    ax3=fig.add_subplot(212)
    yax_lim=[0,100]
    ax3.set_ylabel('Ion concentration (%w)',fontsize=8,fontname='Arial',weight='bold')
    #ax3.grid(True)

    Ions_marker= mpatches.Patch(color=color[a], label=b)
    Ion_legend_properties = {'weight':'bold','size':6}
    legend3= ax3.legend(handles=[Ions_marker],shadow=False,facecolor='w',bbox_to_anchor=(0.14, 0.8-0.05*a),
                       prop=Ion_legend_properties)
    x[b]=color[a]
#Plots can be distingusished based on the authours listed in r_ions

    for u, v in enumerate(r):
       globals()['df_Mean%s' % b] .plot(y=v, kind='line', use_index=True, linestyle='None', markersize=2, ax=ax3,
                  ylim=yax_lim, color=color[a], marker=marker_list[u], markerfacecolor='none',markeredgewidth=.6,fontsize=7,legend=False)

#Legend detrmining authors
legend_properties = {'weight':'regular','size':6.5}
legend = ax3.legend(loc='best', shadow=False, labels=r,facecolor='None',prop=legend_properties)

#plt.figtext(0.9,0.17,"Cl$^-$\n\n\nNa$^+$\n\n\nMg$^{2+}$\n\n\nSO$^{2- 4}$",fontsize=6.5)


fig.tight_layout()  # otherwise the right y-label is slightly clipped

fig.subplots_adjust(wspace=0, hspace=0.0)

fig.savefig('TDS_Ions_Timeseries(2plots_MultiRefrence).png',dpi=1000)
plt.show()
