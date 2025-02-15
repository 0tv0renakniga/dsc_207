import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

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
# squish axes1 into 1d array
axes1 = axes1.flatten()
sns.barplot(y=df.region.value_counts().index,x=df.region.value_counts().values,ax=axes1[0],hue=df.region.value_counts().index)
axes1[0].set_ylabel('Region')
axes1[0].set_xlabel('Number of Countries')
axes1[0].set_title('Distribution of Countries per Region')
sns.boxplot(data=df, x='happiness_score', y='region',ax=axes1[1],hue='region')
axes1[1].set_ylabel('Region')
axes1[1].set_xlabel('Happiness Score')
axes1[1].set_title('Happiness Score per Region')
plt.tight_layout()
#plt.show()

happiness_score_comparison= ['economy_gdp_per_capita', 'family', 'health_life_expectancy', 'freedom','trust_government_corruption', 'generosity', 'dystopia_residual']

def calc_r_sqared(reg,x,y,col):
    x = x.to_numpy()
    y = y.to_numpy()
    coefficents = np.polyfit(x,y,1)
    slope = coefficents[0]
    intercept = coefficents[1]
    line_of_best_fit = slope*x + intercept
    y = y.flatten() 
    line_of_best_fit = line_of_best_fit.flatten()
    # Compute R² using Scikit-Learn
    r2_sklearn = r2_score(y, line_of_best_fit)
    #print(f"R² (Scikit-Learn Calculation): {r2_sklearn}")
    return(r2_sklearn,len(x))

# check for correlations between happiness score and other numeric values
comp_dict_global ={}
for col in happiness_score_comparison:
    x = df.loc[:,col]
    y = df.loc[:,'happiness_score']
    r2, npoints = calc_r_sqared('global',x,y,col)
    key = "_".join(['global',col,'vs_hs'])
    comp_dict_global[key]=(r2,npoints)
# looking globally we don't need to filter for min number of points since observations >=10
comp_dict_global = {i:j[0] for i,j in comp_dict_global.items()}
df_global = pd.DataFrame.from_dict(comp_dict_global, orient='index',columns =['r2_value'])
df_global.index.name ='comparison'
df_global.sort_values('r2_value',ascending=False,inplace=True)
df_global['region'] = [i.split('_')[0] for i in df_global.index.values.tolist()]
df_global['comparison_hs'] = [i.split('_',1)[1].strip('_vs_hs') for i in df_global.index.values.tolist()]
df_global.reset_index(inplace=True,drop=True)

# check for correlations between happiness score and other numeric values by region
comp_dict_regional_points={}
for col in happiness_score_comparison:
    for reg in df.region.unique():
        x = df[df.region==reg].loc[:,col]
        y = df[df.region==reg].loc[:,'happiness_score']
        r2, npoints = calc_r_sqared(reg,x,y,col)
        key = "_".join([reg,col,'vs_hs'])
        comp_dict_regional_points[key]=(r2,npoints)

# take all observations reguardless of number of points
comp_dict_regional = {i:j[0] for i,j in comp_dict_regional_points.items()}

df_regional = pd.DataFrame.from_dict(comp_dict_regional, orient='index',columns =['r2_value'])
df_regional.index.name ='comparison'
df_regional.sort_values('r2_value',ascending=False,inplace=True)
df_regional['region'] = [i.split('_')[0] for i in df_regional.index.values.tolist()]
df_regional['comparison_hs'] = [i.split('_',1)[1].strip('_vs_hs') for i in df_regional.index.values.tolist()]
df_regional.reset_index(inplace=True,drop=True)

#combine regional and global results for r squared comparison for happiness score
df_comp = pd.concat([df_global,df_regional])
df_comp.replace('ealth_life_expectancy', 'health_life_expectancy', inplace=True)
df_comp.sort_values('r2_value', ascending=False, inplace=True)
df_comp.reset_index(drop=True, inplace=True)

# Create the bar chart
plt.figure(figsize=(12, 6))  # Adjust figure size for better readability
sns.barplot(x='region', y='r2_value', hue='comparison_hs', data=df_comp,) #palette='Set3')

plt.ylabel("R-squared Value")
plt.xlabel("Region - Comparison")
plt.xticks(rotation=45)
plt.title("R-squared Values for Happiness Factor Comparisons")
plt.tight_layout() # Adjust layout to prevent labels from overlapping
#plt.show()

# now only looking at regions with enough data(n>=10)
# note this cuts our observations in half 70 => 35
comp_dict_regional = {i:j[0] for i,j in comp_dict_regional_points.items() if j[1] >=10}
df_regional = pd.DataFrame.from_dict(comp_dict_regional, orient='index',columns =['r2_value'])
df_regional.index.name ='comparison'
df_regional.sort_values('r2_value',ascending=False,inplace=True)
df_regional['region'] = [i.split('_')[0] for i in df_regional.index.values.tolist()]
df_regional['comparison_hs'] = [i.split('_',1)[1].strip('_vs_hs') for i in df_regional.index.values.tolist()]
df_regional.reset_index(inplace=True,drop=True)

#combine regional and global results for r squared comparison for happiness score
df_comp = pd.concat([df_global,df_regional])
df_comp.replace('ealth_life_expectancy', 'health_life_expectancy', inplace=True)
df_comp.sort_values('r2_value', ascending=False, inplace=True)
df_comp.reset_index(drop=True, inplace=True)

# Create the bar chart
plt.figure(figsize=(12, 6))  # Adjust figure size for better readability
sns.barplot(x='region', y='r2_value', hue='comparison_hs', data=df_comp,) #palette='Set3')
plt.ylabel("R-squared Value")
plt.xlabel("Region - Comparison")
plt.xticks(rotation=45)
plt.title("R-squared Values for Happiness Factor Comparisons(Regions with n>=10)")
plt.tight_layout() # Adjust layout to prevent labels from overlapping
#plt.show()

'''
objective 2: To what extent does economic development (GDP per capita) influence
happiness levels globally(1), and how does this relationship interact with other factors 
like health(2)? Are there diminishing returns to economic growth in 
terms of happiness(3)?
- (1) gdp vs happiness score
- (2) gdp vs family and gdp vs health life expectancy
- (3) look at plot from (1) does happiness plateau at a high enough gdp?
'''
# (1) gdp vs happiness score
df_global.replace('ealth_life_expectancy', 'health_life_expectancy', inplace=True)
# plots
fig2, axes2 = plt.subplots(2,1,figsize=(15,10))
# squish axes2 into 1d array
axes2 = axes2.flatten()
sns.barplot(x=df_global.r2_value,y=df_global.region,ax=axes2[0],hue=df_global.comparison_hs)
axes2[0].set_ylabel('Globally')
axes2[0].set_xlabel('R-squared Value')
axes2[0].set_title('Metric Influence on Happiness Score Globally')
sns.scatterplot(data=df, x='economy_gdp_per_capita', y='happiness_score',ax=axes2[1],hue='region')
axes2[1].set_ylabel('Happiness Score')
axes2[1].set_xlabel('GDP per Capita')
axes2[1].set_title('Happiness Score vs GDP per Capita')
plt.tight_layout()
#plt.show()

#(2) gdp vs family and gdp vs health life expectancy
# Calculate the correlation matrix for GDP and other numeric columns
correlation_matrix = df[['economy_gdp_per_capita', 'happiness_score', 'health_life_expectancy', 'family', 'freedom', 'generosity', 'trust_government_corruption']].corr()
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
tick_labels = ['GDP', 'Happiness', 'Health', 'Family', 'Freedom', 'Generosity', 'Trust']
heatmap.set_xticklabels(tick_labels, rotation=45)
heatmap.set_yticklabels(tick_labels, rotation=0)
plt.title('Correlation Heatmap: GDP and Numeric Metrics')
plt.tight_layout()
#plt.show()

# (3) look at plot from (1) does happiness plateau at a high enough gdp?
fig3, axes3 = plt.subplots(1,2,figsize=(12, 6), sharey=True)  # Adjust figure size for better readability
axes3 = axes3.flatten()
# Plot GDP per Capita < 1.45 vs. Happiness Score
sns.scatterplot(x='economy_gdp_per_capita', y='happiness_score', data=df.query('economy_gdp_per_capita < 1.45'),ax=axes3[0],alpha=0.5) 
sns.regplot(x='economy_gdp_per_capita', y='happiness_score', data=df.query('economy_gdp_per_capita < 1.45'),ax=axes3[0], scatter=False, color='red')
X1 = df.query('economy_gdp_per_capita < 1.45').economy_gdp_per_capita.values.reshape(-1,1)
y1 = df.query('economy_gdp_per_capita < 1.45').happiness_score.values
model_1 = LinearRegression().fit(X1, y1)
r_squared_1 = model_1.score(X1, y1)
slope_1 = model_1.coef_[0]
intercept_1 = model_1.intercept_

# Display R-squared value and line of best fiton the plot
axes3[0].text(0.05, 0.95, f'R² = {r_squared_1:.2f}\n y={slope_1:.2f}x + {intercept_1:.2f}', transform=axes3[0].transAxes, fontsize=12, verticalalignment='top')
axes3[0].set_ylabel("Happiness Score")
axes3[0].set_xlabel("GDP per Capita")
axes3[0].set_title("GDP per Capita <1.45")

# Plot GDP per Capita >= 1.45 vs. Happiness Score
sns.scatterplot(x='economy_gdp_per_capita', y='happiness_score', data=df.query('economy_gdp_per_capita >= 1.45'),ax=axes3[1],alpha=0.5) 
sns.regplot(x='economy_gdp_per_capita', y='happiness_score', data=df.query('economy_gdp_per_capita >= 1.45'),ax=axes3[1], scatter=False, color='red')
X2 = df.query('economy_gdp_per_capita >= 1.45').economy_gdp_per_capita.values.reshape(-1,1)
y2 = df.query('economy_gdp_per_capita >= 1.45').happiness_score.values
model_2 = LinearRegression().fit(X2, y2)
r_squared_2 = model_2.score(X2, y2)
slope_2 = model_2.coef_[0]
intercept_2 = model_2.intercept_
# Display R-squared value on the plot
axes3[1].text(0.05, 0.95, f'R² = {r_squared_2:.2f}\n y={slope_2:.2f}x + {intercept_2:.2f}', transform=axes3[1].transAxes, fontsize=12,verticalalignment='top')
#axes3[1].set_ylabel("Happiness Score")
axes3[1].set_xlabel("GDP per Capita")
axes3[1].set_title("GDP per Capita >=1.45")
plt.tight_layout() # Adjust layout to prevent labels from overlapping
#plt.show()

'''
objective 3: How do trust in government and perceived freedom influence happiness levels(1), 
and are these factors more important in certain regions(2)? Is there a 
link between trust and happiness(3)?
- (1) trust in govt vs. happiness score and freedom vs. happiness (globally)
- (2) trust in govt vs. happiness score and freedom vs. happiness (region)
- (3) trust in govt corruption vs. happiness similarities between regions?
'''
# (1) trust in govt vs. happiness score and freedom vs. happiness (globally)
# plots for comparison btwn trust in govt and freedom vs happiness score
fig4, axes4 = plt.subplots(1,2,figsize=(12, 6), sharey=True)  # Adjust figure size for better readability
axes4 = axes4.flatten()
# trust vs. Happiness Score
sns.scatterplot(x='trust_government_corruption', y='happiness_score', data=df,ax=axes4[0],alpha=0.5) 
sns.regplot(x='trust_government_corruption', y='happiness_score', data=df,ax=axes4[0], scatter=False, color='red')
X1 = df.trust_government_corruption.values.reshape(-1,1)
y1 = df.happiness_score.values
model_1 = LinearRegression().fit(X1, y1)
r_squared_1 = model_1.score(X1, y1)
slope_1 = model_1.coef_[0]
intercept_1 = model_1.intercept_

# Display R-squared value and line of best fiton the plot
axes4[0].text(0.05, 0.95, f'R² = {r_squared_1:.2f}\n y={slope_1:.2f}x + {intercept_1:.2f}', transform=axes4[0].transAxes, fontsize=12, verticalalignment='top')
axes4[0].set_ylabel("Happiness Score")
axes4[0].set_xlabel("Trust in Government")
axes4[0].set_title("How Trust in Government Influences Happiness Score")

# Plot freedom vs. Happiness Score
sns.scatterplot(x='freedom', y='happiness_score', data=df,ax=axes4[1],alpha=0.5) 
sns.regplot(x='freedom', y='happiness_score', data=df,ax=axes4[1], scatter=False, color='red')
X2 = df.freedom.values.reshape(-1,1)
y2 = df.happiness_score.values
model_2 = LinearRegression().fit(X2, y2)
r_squared_2 = model_2.score(X2, y2)
slope_2 = model_2.coef_[0]
intercept_2 = model_2.intercept_
# Display R-squared value on the plot
axes4[1].text(0.05, 0.95, f'R² = {r_squared_2:.2f}\n y={slope_2:.2f}x + {intercept_2:.2f}', transform=axes4[1].transAxes, fontsize=12,verticalalignment='top')
axes4[1].set_xlabel("Freedom")
axes4[1].set_title("How Freedom Influences Happiness Score")
plt.tight_layout() # Adjust layout to prevent labels from overlapping
#plt.show()

# (2) trust in govt vs. happiness score and freedom vs. happiness (region)
# (3) trust in govt corruption vs. happiness similarities between regions?
# Calculate cor matrix for trust in government and freedom vs. happiness score by region
# note not all regions have enough data to calculate a correlation, so we will ignore north america and australia and new zealand
fig_asia, axes_asia = plt.subplots(3,1,figsize=(15,10),sharex=True)
fig_asia.suptitle('Heatmaps for Asia')
axes_asia = axes_asia.flatten()
fig_europe, axes_europe = plt.subplots(2,1,figsize=(15,10),sharex=True)
fig_europe.suptitle('Heatmaps for Europe')
axes_europe = axes_europe.flatten()
fig_africa, axes_africa = plt.subplots(2,1,figsize=(15,10),sharex=True)
fig_africa.suptitle('Heatmaps for Africa')
axes_africa = axes_africa.flatten() 
fig_la, axes_la = plt.subplots(1,1,figsize=(15,10))

regions_axes = [
    (['Sub-Saharan Africa','Middle East and Northern Africa'], axes_africa),
    (['Latin America and Caribbean'], [axes_la]),
    (['Western Europe','Central and Eastern Europe'], axes_europe),
    (['Southern Asia','Eastern Asia','Southeastern Asia'], axes_asia),
]

for regs, axes in regions_axes:
    for i,reg in enumerate(regs):
        correlation_matrix = df[df.region == reg][['trust_government_corruption', 'happiness_score', 'freedom']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[i])
        tick_labels = ['Trust in Government', 'Happiness Score', 'Freedom']
        axes[i].set_xticklabels(tick_labels, rotation=45)
        axes[i].set_yticklabels(tick_labels, rotation=0)
        axes[i].set_title(f'Correlation Heatmap: {reg}')
plt.tight_layout()
plt.show()
for reg in df.region.unique():
    correlation_matrix = df[df.region==reg][['trust_government_corruption', 'happiness_score', 'freedom']].corr()
    print(f"Region: {reg}")
    print(correlation_matrix)
    print(df[df.region==reg].shape[0])
    print()
#correlation_matrix = df[['trust_government_corruption', 'happiness_score', 'freedom']].corr()
#print(correlation_matrix)
#- (3) trust in govt corruption vs. happiness (top 5 and bottom 5)