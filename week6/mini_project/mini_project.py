import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def rename_columns(df):
    '''
    Renames pandas df column names for 2016.csv

    Input
    -----
    df: pd.Dataframe

    Returns
    -------
    df: pd.Dataframe
    '''

    new_col_names = {i:i.lower().replace(' ','_').replace('(','').replace(')','') for i in df.columns}

    return(df.rename(columns=new_col_names))

def clean_df_values(df):
    '''
    Remove null values and format observations to appropriate data 
    types for2016.csv

    Input
    -----
    df: pd.Dataframe

    Returns
    -------
    df: pd.Dataframe
    '''
    pass

def create_df_by_region(df):
    '''
    create new dataframes with variable names corresponding to each unique
    region. the following will be created:

    western_europe_df, north_america_df, australia_and_new_zealand_df,
    middle_east_and_northern_africa_df, latin_america_and_caribbean_df,
    southeastern_asia_df, central_and_eastern_europe_df,
    eastern_asia_df, sub_saharan_africa_df, southern_asia_df

    Input
    -----
    df: pandas.Dataframe

    Returns
    -------

    '''
    # create a list of unique regions
    regions = df.region.unique()

    new_region_df_names = [i.lower() for i in regions]
    new_region_df_names = [i.replace(' ', '_') for i in new_region_df_names]
    new_region_df_names = [i.replace('-', '_') for i in new_region_df_names]
    print(new_region_df_names)

    for df_name,region in zip(new_region_df_names,regions):
        globals()[f'{df_name}_df'] = df[df['region'].isin([region])]

    print(globals().keys())
def main():
    # read 2016.csv supplied for mini project
    df = pd.read_csv('2016.csv')
    
    # rename column names to access column as df property
    df = rename_columns(df)
    
    create_df_by_region(df)
    print(north_america_df.describe())

main()
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
