import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# read 2016.csv
df = pd.read_csv('2016.csv')
print(df.info())
# create new col names
new_col_names = {i:i.lower().replace(' ','_').replace('(','').replace(')','') for i in df.columns}

# assign new col names
df.rename(columns=new_col_names, inplace=True)

# remove countries that have a zero for an observation
rows_with_zero = df[(df == 0).any(axis=1)]
'''
list of countries that will be removed from df 'Somalia' 'Bosnia and Herzegovina' 'Greece' 
'Sierra Leone' 'Sudan' 'Togo'
'''
country_with_zero = rows_with_zero.country.unique()

# df with observations that contain a 0 are removed
df = df[~df.country.isin(country_with_zero)]

# set region and country to strings
df.region = df.region.astype(str)
df.country = df.country.astype(str)

# reset happiness index to account for removing countries
df.sort_values(by='happiness_score', inplace=True, ascending=False)
df.happiness_rank = df.happiness_score.rank(method='max', ascending=False).astype(int)
df.reset_index(drop=True,inplace=True)

'''
define objectives
objective 1: How does happiness vary across different regions, and what are the 
primary factors contributing to these regional differences? Can we identify 
specific combinations of economic, social, and political factors that explain
high or low happiness scores in particular regions?

objective 2: To what extent does economic development (GDP per capita) influence
happiness levels globally, and how does this relationship interact with other factors 
like health and social support? Are there diminishing returns to economic growth in 
terms of happiness?

objective 3: How do trust in government and perceived freedom influence happiness levels, 
and are these factors more important in certain regions or cultural contexts? Is there a 
link between corruption perception and happiness?
'''