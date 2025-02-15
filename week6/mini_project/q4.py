import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# read 2016.csv
df = pd.read_csv('2016.csv')

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
objective 1: How does happiness vary across different regions(1), and what are the 
primary factors contributing to these regional differences(2)? Can we identify 
specific combinations of economic, social, and political factors that explain
high or low happiness scores in particular regions?
- (1) max, min, mean,median for each region
- (2) plot col vs happiness score for each region and select col with highest
r squared value. highest r squared means this col has highest impact on
happiness score
'''
# sort by number of countries in each region
df['sort_col'] = df['region'].map(df['region'].value_counts())
df = df.sort_values(by='sort_col', ascending=False).drop('sort_col', axis=1)

# plots
fig1, axes1 = plt.subplots(2,1,figsize=(15,10))
axes1 = axes1.flatten()
#fig1.suptitle('Happiness Score by Region')
#sns.barplot(y=countries_by_region.index,x=countries_by_region.values,ax=axes1[0],hue=countries_by_region.index)
sns.barplot(y=df.region.value_counts().index,x=df.region.value_counts().values,ax=axes1[0],hue=df.region.value_counts().index)
axes1[0].set_ylabel('Region')
axes1[0].set_xlabel('Number of Countries')
axes1[0].set_title('Distribution of Countries per Region')
sns.boxplot(data=df, x='happiness_score', y='region',ax=axes1[1],hue='region')
axes1[1].set_ylabel('Region')
axes1[1].set_xlabel('Happiness Score')
axes1[1].set_title('Happiness Score per Region')
plt.tight_layout()
plt.show()
#print(df.columns)
happiness_score_comparison= ['economy_gdp_per_capita', 'family', 'health_life_expectancy', 'freedom','trust_government_corruption', 'generosity', 'dystopia_residual']

for reg in df.region.unique():
    print(df[df.region==reg].economy_gdp_per_capita.shape)
'''
for col in happiness_score_comparison:
    plt.figure()
    sns.scatterplot(data=df,x=col,y='happiness_score')
    plt.tight_layout()
    plt.show()
'''
'''
objective 2: To what extent does economic development (GDP per capita) influence
happiness levels globally(1), and how does this relationship interact with other factors 
like health and social support(2)? Are there diminishing returns to economic growth in 
terms of happiness(3)?
- (1) gdp vs happiness score
- (2) gdp vs family and gdp vs health life expectancy
- (3) look at plot from (1) does happiness plateau at a high enough gdp?
'''

'''
objective 3: How do trust in government and perceived freedom influence happiness levels(1), 
and are these factors more important in certain regions or cultural contexts(2)? Is there a 
link between corruption perception and happiness(3)?
- (1) trust in govt vs. happiness score and freedom vs. happiness (globally)
- (2) trust in govt vs. happiness score and freedom vs. happiness (region)
- (3) trust in govt corruption vs. happiness (top 5 and bottom 5)
'''