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

# get total records and fields for dataset
rec = df.shape[0]
fields = df.shape[1]

# get total countries and regions
unique_country_total = df.country.nunique()
unique_regions_total = df.region.nunique()
unique_regions = ', '.join(df.region.unique())

# get metrics for each country
num_data = df.select_dtypes(include='number')
metrics = ", ".join(num_data.columns)
q1=f'''
1. High Level View 
The 2016.csv contains {fields} fields, and {rec} records.
The dataset gives us {unique_country_total} unique countries, that are split across 
{unique_regions_total} different regions. These unique regions are the following:
{unique_regions}. We are given the following metrics for each country: {metrics}.
This data set could be useful for looking at how government corruption or freedom varies
by region and country. It could also be useful for determining which of the metrics given
is most influential on happiness score.
'''
import io

def info_to_dataframe(df):
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue().split('\n')
    
    data = []
    for line in info_str:#[5:-3]:  # Skip header and footer lines
        #print([i for i in line.split(' ')[1:] if len(i) > 0])
        parts = line.split(maxsplit=2)
        data.append(parts)
    data = data[5:-3]
    for i in data:
        i[-1] = i[-1].strip()
        non_null = " ".join(i[-1].split(' ')[:2])
        d_type = i[-1].split(' ')[-1]
        clean = [i[1], non_null, d_type]
        print(clean)

    info_df = pd.DataFrame(data, columns=['col', 'Non-Null Count', 'Dtype'])
    return info_df

null_table = info_to_dataframe(df)
#print(null_table.col.values)
