# ------------------------------------------------------
# Code developer: Somayeh Sima
# Creation date: 8/10/2019
# This code:
# 1. reads a csv data file into a Pandas DataFrame,
# 2. calculates correlation matrix of desired fields of data and saves as a csv file,
# 3. plots desired fields of data as a matrix plot with two different symbols before and after a desired breakpoint.
# ------------------------------------------------------
# Import the Pandas and Matplotlib packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# Read the CSV file into a Pandas data frame object
df = pd.read_csv('Ul_AgEnvi_Dataset(1995_2015).csv', header=1, sep=',',
                 low_memory=False)

# Get the number of values in the subset dataframe to check the data
print('There are ' + str(len(df)) + ' data points in your subset.')

# Get a list of columns in the dataframe
lcolumn=list(df.columns)
#print(lcolumn)

## Divide the dataset into two periods
#Changing_Point=input('insert the year you want to divide data set into before and after that:')
Changing_Point=2012

# apply the filter on year column
Years=list(df['Year'])

# add a list called Period to the current dataframe
Period=[]
for i,y in enumerate(Years):
  if y<int(Changing_Point):
    Period.insert(i,'Before %s'% Changing_Point)
  else:
    Period.insert(i,'%s and after'%Changing_Point)
df["Period"]=Period


## Data slicing : Determine the target columns for the matrix plot
#insert numbers of the desired columns of dataframe to be plotted two by two

#des_columns=[int(x) for x in input('insert column numbers you want to be plotted(Default:14,13,15,17,22):').split(',')]
des_columns=[14,13,15,17,22]
plotno=len(des_columns)
# Add the Period column to the desired columns to be able to filter plot by periods
des_columns.append(len(lcolumn))
des_df=df.iloc[:,des_columns]

#Divide the dataset into two data sets: pre and post of the changing point
des_df1=des_df[des_df['Period']=='Before %s'%(Changing_Point)]
des_df2=des_df[des_df['Period']=='%s and after'%(Changing_Point)]

# Plots a matrix scatter plot using the Seaborn library
sns.set(style="ticks")

# Set a color scheme for palette

# Option 1: set 2 color schemes as RGB topple
# To convert RGB in 255 basis to values in the range of [0,1] divide each number to 255
Mplot=sns.pairplot(des_df,hue="Period",dropna='True',markers=["s","o"],height=3, aspect=1,diag_kind='auto',
                  plot_kws=dict(s=100,linewidth=2,alpha=.9),palette={"Before 2012":(0.439,0.678,0.278),"2012 and after":(0.220,0.341,0.137)},
                 diag_kws=dict(shade='True'))

# Option 1: set 2 color in a continuous color scheme
# colors = sns.color_palette("Greens", 2)
#Mplot=sns.pairplot(des_df,hue="Period",dropna='True',markers=["s","o"],height=3, aspect=1,diag_kind='auto',
#                  plot_kws=dict(s=100,linewidth=2,alpha=.9),palette=colors,
#                 diag_kws=dict(shade='True'))

# define a function to hide either the lower or upper part of the plot
def hide_current_axis(*args, **kwds):
  plt.gca().set_visible(False)


#hide_updiag=input('Do you want to hide the upper diagonal plots(Y/N)?')
hide_updiag='Y'

if hide_updiag=='Y':
    Mplot.map_upper(hide_current_axis)
    Mplot.map_upper(hide_current_axis)
    #Mplot.map_diag(hide_current_axis)

# Adjusts the legend and its position
Mplot._legend.get_title().set_fontsize(16)
Mplot._legend.get_title().set_weight('bold')
Mplot._legend.set_bbox_to_anchor((0.95, .36))


# Multiple line axes labels
# This function get a string and the order of space from which you want to break the string into multiple lines
# for example: " Hello my world" and 2 returns : "Hello my \n world"
def multiline_str(s,space_break_No):
    s_pos=[i for i, letter in enumerate(s) if letter == ' ']
    if len(s_pos)!=space_break_No:
        s=s[:s_pos[1]]+'\n'+s[s_pos[1]:]
    return s

#Customize axes labels
for ax in plt.gcf().axes:
    leg=ax.get_legend()
    xl=ax.get_xlabel()
    yl = ax.get_ylabel()
    if xl!='':
        xl=multiline_str(xl,2)
    if yl!= '':
        yl=multiline_str(yl, 2)

    ax.set_xlabel(xl,fontsize=16,weight='bold')
    ax.set_ylabel(yl,fontsize=16,weight='bold')
    #leg.label(fontsize='large',weight='bold')

# Calculates the correlation matrix of  the desired datasets and exports it as a csv file
cc=des_df.corr()
cc_p1=des_df1.corr()
cc_p2=des_df2.corr()

cc.to_csv('CC of UL_AgEnvi_dataset.csv')
cc_p1.to_csv('CC of UL_AgEnvi_dataset_Before %s'%(Changing_Point)+'.csv')
cc_p2.to_csv('CC of UL_AgEnvi_dataset_After %s'%(Changing_Point)+'.csv')

#status=input('select the data set for regression stat calculation(whole,pre,post):')
choice={'whole':(des_df,'black'),'pre':(des_df1,(0.439,0.678,0.278)),'post':(des_df2,(0.22,0.341,.137))}
des_ds_pre =choice['pre'][0]
des_color_pre=choice['pre'][1]
des_ds_post =choice['post'][0]
des_color_post=choice['post'][1]

#ldes_column=list(des_ds.columns)
#Calculate p-value matrix
# create empty arrays to store r, p-value and n values
pval_pre = np.zeros([des_df.shape[1]-1,des_df.shape[1]-1])
rval_pre = np.zeros([des_df.shape[1]-1,des_df.shape[1]-1])
N_pre=np.zeros([des_ds_pre.shape[1]-1,des_df.shape[1]-1])

pval_post = np.zeros([des_df.shape[1]-1,des_df.shape[1]-1])
rval_post = np.zeros([des_df.shape[1]-1,des_df.shape[1]-1])
N_post=np.zeros([des_df.shape[1]-1,des_df.shape[1]-1])

for i in range(des_ds_pre.shape[1]-1):
    for j in range(des_ds_pre.shape[1]-1):
         des_ds0_pre= des_ds_pre.iloc[:, [i, j]].dropna()
         rp = stats.pearsonr(des_ds0_pre.iloc[:, 0], des_ds0_pre.iloc[:, 1])
         N_pre[i, j] = len(des_ds0_pre.iloc[:, 0])  # n is the number of pair data points
         rval_pre[i, j] = rp[0]
         pval_pre[i, j] = rp[1]
         # print(i, j, rp,(n), ldes_column[i], ldes_column[j],des_ds0)

for i in range(des_ds_post.shape[1]-1):
    for j in range(des_ds_post.shape[1]-1):
         des_ds0_post= des_ds_post.iloc[:, [i, j]].dropna()
         rp = stats.pearsonr(des_ds0_post.iloc[:, 0], des_ds0_post.iloc[:, 1])
         N_post[i, j] = len(des_ds0_post.iloc[:, 0])  # n is the number of pair data points
         rval_post[i, j] = rp[0]
         pval_post[i, j] = rp[1]

#Adjust the text position on plots for a 5-subplot and 4-subplot matrix
if plotno==5:
    dis=0.19
    xin=0.08
    yin=0.973
    vdis=0.034
else:
    xin=0.097
    dis=0.235
    yin=0.96
    vdis=0.047

for i in range (1,len(cc_p1)):
    for j in range(0,i):
        r=[rval_pre[i,j],rval_post[i,j]]
        p=[pval_pre[i,j],pval_post[i,j]]
        n=[N_pre[i,j],N_post[i,j]]

        # Adding regression parameters on the plots
        plt.figtext(xin+j*dis,yin-vdis-i*dis,'n=%i\nr=%0.2f \np=%0.3f'%(n[0],r[0],p[0]),fontsize=12,
                    fontweight='bold',color=des_color_pre)
        plt.figtext( xin+j*dis,yin-i*dis,'n=%i\nr=%0.2f \np=%0.3f'%(n[1],r[1],p[1]),fontsize=12,
                    fontweight='bold',color=des_color_post)

plt.show()
plt.savefig("UL_AgVariables_Matrixplot.png",bbox_inches = "tight",dpi=500)

