import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# read 2016.csv
df = pd.read_csv('2016.csv')

# create new col names
new_col_names = {i:i.lower().replace(' ','_').replace('(','').replace(')','') for i in df.columns}

# assign new col names
df.rename(columns=new_col_names, inplace=True)

null_values_col = df.isnull().sum()
null_values_sum = null_values_col.values.sum()
print(df.dtypes)
obs = df.shape[0]
fields = df.shape[1]
num_data = df.select_dtypes(include='number')
num_data_total = num_data.shape[1]
num_vars = "\n".join(num_data.columns)
cat_data = df.select_dtypes(include='object')
cat_data_total = cat_data.shape[1]
cat_vars = "\n".join(cat_data.columns)
q1=f'''
1. High Level View 
The 2016.csv contains {null_values_sum} null values, {fields} fields, and {obs} observations.
The dataset contains {cat_data_total} categorical variables and {num_data_total} numerical variables.

The categorical variables are the following:
{cat_vars}

The numerical variables are the following:
{num_vars}



'''
print(q1)
def rename_columns(df):
    '''
    Renames pandas df column names for 2016.csv
    '''
    new_col_names = {i:i.lower().replace(' ','_').replace('(','').replace(')','') for i in df.columns}

    return(df.rename(columns=new_col_names))

def clean_df_values(df):
    '''
    Remove null values and format observations to appropriate data 
    types for2016.csv
    '''
    # find observations with 0
    rows_with_zero = df[(df == 0).any(axis=1)]
    # list of countries that will be removed from df
    # 'Somalia' 'Bosnia and Herzegovina' 'Greece' 
    # 'Sierra Leone' 'Sudan' 'Togo'
    country_with_zero = rows_with_zero.country.unique()

    # new df with observations that contain a 0 are removed
    df_removed_zeros = df[~df.country.isin(country_with_zero)]

    return(df_removed_zeros)

def create_df_by_region(df):
    '''
    create new dataframes with variable names corresponding to each unique
    region. the following will be created:

    df_western_europe, df_north_america, df_australia_and_new_zealand, 
    df_middle_east_and_northern_africa, df_latin_america_and_caribbean, 
    df_southeastern_asia, df_central_and_eastern_europe, df_eastern_asia, 
    df_sub_saharan_africa, df_southern_asia


    Input
    -----
    df: pandas.Dataframe

    Returns
    -------

    '''
    # create a list of unique regions
    regions = df.region.unique()

    # format region names
    new_region_df_names = [i.lower() for i in regions]
    new_region_df_names = [i.replace(' ', '_') for i in new_region_df_names]
    new_region_df_names = [i.replace('-', '_') for i in new_region_df_names]

    for df_name,region in zip(new_region_df_names,regions):
        globals()[f'df_{df_name}'] = df[df['region'].isin([region])]


def main():
    # read 2016.csv supplied for mini project
    df = pd.read_csv('2016.csv')
    
    # rename column names to access column as df property
    df = rename_columns(df)
    
    create_df_by_region(df)
    df_regions = [df_western_europe, df_north_america, df_australia_and_new_zealand, df_middle_east_and_northern_africa, df_latin_america_and_caribbean, df_southeastern_asia, df_central_and_eastern_europe, df_eastern_asia, df_sub_saharan_africa, df_southern_asia]

    print(df.dtypes)
    print(df.shape)
    df_no_zeros =clean_df_values(df)
    print(df.isnull().sum())

    

#main()


'''
useful pandas commands
top_n = 10

top_10_neighbourhoods = listings_df.neighbourhood.value_counts().iloc[0:top_n]

filtered_df = listings_df[listings_df['neighbourhood'].isin(top_10_neighbourhoods.index)]
'''

'''
useful seaborn subpots

fig, axes = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle('Subplots created from part 4')
sns.barplot(x=top_10_neighbourhoods.index,
            y=top_10_neighbourhoods.values,
            ax=axes[0])
axes[0].tick_params(axis='x', labelrotation=90)
axes[0].set_xlabel('Neighbourhood')
axes[0].set_ylabel('Count')
axes[0].set_title('Top 10 Neighbourhoods')
sns.boxplot(y='price',
            x='neighbourhood',
            data=filtered_df,
            hue='neighbourhood',
            showfliers = False,
            ax=axes[1])
axes[1].tick_params(axis='x', labelrotation=90)
axes[1].set_xlabel('Neighbourhood')
axes[1].set_ylabel('Price')
axes[1].set_title('Prices of Listings by Neighbourhood')
'''


def calc_bins(price_range,samples,stdev):
  '''
  calc optimal number of histogram bins based on Scott 1979
  https://academic.oup.com/biomet/article-abstract/66/3/605/232642
  bins =R(n^(1/3))/(3.49σ)
  R: range of data
  n: sample size
  σ: standard deviation of data
  '''
  bins = price_range*(samples**(1/3))/(3.49*stdev)
  return(int(bins))
