import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# read 2016.csv
df = pd.read_csv('2016.csv')

# create new col names
new_col_names = {i:i.lower().replace(' ','_').replace('(','').replace(')','') for i in df.columns}

# assign new col names
df.rename(columns=new_col_names, inplace=True)

# check for null values
null_values_col = df.isnull().sum()
null_values_sum = null_values_col.values.sum()

# get numerical and categorical data
num_data = df.select_dtypes(include='number')
num_data_total = num_data.shape[1]
num_vars = "\n".join(num_data.columns)
cat_data = df.select_dtypes(include='object')
cat_data_total = cat_data.shape[1]
cat_vars = "\n".join(cat_data.columns)

# observations with 0
rows_with_zero = df[(df == 0).any(axis=1)]
zero_obs_total = rows_with_zero.shape[0]

# drop region column and set country as the index of the dataframe
rows_with_zero.drop(columns='region',inplace=True)
rows_with_zero.set_index('country', drop=True, inplace=True)

# find what metrics are 0 and the corresponding country
country_metric_list =[]
rows, cols = np.where(rows_with_zero == 0)
for row, col in zip(rows, cols):
    country_metric_list.append(f"Country: {rows_with_zero.index[row]}, Metric: {rows_with_zero.columns[col]}")
country_metric_str = '\n'.join(country_metric_list)

# get trends for numeric values
def get_trends(df):
    trends=[]
    for col in df.columns:
        trends.append(f"{col.upper()}\nmean:{df[col].mean():5.3g}\tmedian:{df[col].median():5.3g}\tstandard deviation: {df[col].std():5.3g}\n")
    return("\n".join(trends))
# not intrested in trend for happiness rank so drop it from the numeric data    
num_data.set_index('happiness_rank', drop=True, inplace=True)
trends=get_trends(num_data)

# show distribution of numerical data
def calc_bins(col_range,samples,stdev):
  '''
  calc optimal number of histogram bins based on Scott 1979
  https://academic.oup.com/biomet/article-abstract/66/3/605/232642
  bins =R(n^(1/3))/(3.49σ)
  R: range of data
  n: sample size
  σ: standard deviation of data
  '''
  bins = col_range*(samples**(1/3))/(3.49*stdev)
  return(int(bins))
  
rows = 2
columns = 5
fig, axes = plt.subplots(rows,columns,figsize=(10,5))
# flatten axes to itterate through columns
axes = axes.flatten()
fig.suptitle('DISTRIBUTION OF NUMERICAL DATA')

for i, col in enumerate(num_data.columns):
    col_range = (num_data[col].max() -num_data[col].min())
    samples = num_data[col].shape[0]
    stdev = num_data[col].std()
    bins = calc_bins(col_range,samples,stdev)
    sns.histplot(data=num_data, x=col, ax=axes[i],bins=bins)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('count')

plt.tight_layout()
plt.show()
# hisogram of happiness score by region for top 3 happiest regions
avg_by_region = df.groupby('region').happiness_score.mean()
avg_by_region.sort_values(inplace=True,ascending=False)
top_3_regions = list(avg_by_region[0:3].index)

df_top_3_regions = df[df.region.isin(top_3_regions)]
plt.figure()
sns.histplot(data=df_top_3_regions,x='happiness_score',hue='region')
plt.title('TOP 3 HAPPIEST REGIONS')
plt.tight_layout()
plt.show()

'''
for i, col in enumerate(num_data.columns):
    sns.histplot(data=num_data,x=col,ax=axes[i])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
plt.show()
'''    
q2=f'''
2. Preliminary Exploration 
The dataset contains {null_values_sum} null values, {cat_data_total} categorical variables, and {num_data_total} numerical variables.

The categorical variables are the following:
{cat_vars}

The numerical variables are the following:
{num_vars}

Are there quality issues in the dataset?
There are no null values in this dataset. However, there are {zero_obs_total} 
observations that are zero. Below are the countries and the corresponding metric they are missing:
{country_metric_str}

What will you need to clean and/or transform the raw data for analysis?
(i) Since the data type for our categorical columns is an 'object' dtype we will
convert these columns to a string data type. 

(ii) We will also need to drop the countries with an observation of 0.

(iii) The happiness_rank is an integer value that corresponds to the happiness_score, such that the max 
happiness is given a happiness_rank of 1, the min happiness is given a happiness_rank of n where n is the 
total number of records.Therefore, we need to reset the happiness rank to account for these dropped countries with an observation of 0. 

What are trends in the dataset and distribution of numerical data?
(i) The trends are as follows:
{trends}
(ii) Distribution of numerical data is shown below:

Preliminary Exploration Task 1: Check for null values
Preliminary Exploration Task 2: Histogram of happiness score for top 3 regions
'''
print(q2)

