import pandas as pd
import numpy as np
import pycountry
import pycountry_convert

from sklearn.mixture import GaussianMixture

from sklearn.neighbors import LocalOutlierFactor, KernelDensity, KDTree


from math import sqrt, log, exp, floor
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
from matplotlib import style
matplotlib.style.use('seaborn-v0_8')
gdp_df = pd.read_csv('info.csv')
print(gdp_df.head())
full_countries_list = gdp_df.ISO3.unique()
print(gdp_df.columns)

df_mv = gdp_df.iloc[:,1:(gdp_df.shape[1]-3)]

print('% of NaN cells: '+str(
    df_mv.groupby('Year').apply(lambda x: x.isnull().sum()).iloc[:,1:].sum().sum()/(df_mv.iloc[:,1:].shape[0]*df_mv.iloc[:,1:].shape[1])))

start_year = 2001
end_year = 2020


gdp_df.insert(2, 'gdp_real_gwt_prev', gdp_df.groupby(by="ISO3").shift(1)['gdp_real_gwt'], 'gdp_real_gwt_prev')
gdp_df.insert(2, 'gdp_real_gwt_next', gdp_df.groupby(by="ISO3").shift(-1)['gdp_real_gwt'], 'gdp_real_gwt_next')
gdp_df.insert(2, 'pop_growth_next', gdp_df.groupby(by="ISO3").shift(-1)['pop_growth'], 'pop_growth_next')


gdp_df = gdp_df[gdp_df['Year']>=start_year]
gdp_df = gdp_df[gdp_df['Year']<=end_year]


gdp_df = gdp_df.loc[gdp_df.groupby('ISO3')['gdp_real_gwt_next'].filter(lambda x: len(x[pd.isnull(x)] ) < 1).index,:]
gdp_df = gdp_df.loc[gdp_df.groupby('ISO3')['gdp_real_gwt'].filter(lambda x: len(x[pd.isnull(x)] ) < 1).index,:]
gdp_df = gdp_df.loc[gdp_df.groupby('ISO3')['gdp_real_gwt_prev'].filter(lambda x: len(x[pd.isnull(x)] ) < 1).index,:]
gdp_df = gdp_df.loc[gdp_df.groupby('ISO3')['gdp_pp_govt'].filter(lambda x: len(x[pd.isnull(x)] ) < len(x)).index,:]
gdp_df = gdp_df.loc[gdp_df.groupby('ISO3')['gdp_pp_private'].filter(lambda x: len(x[pd.isnull(x)] ) < len(x)).index,:]
gdp_df = gdp_df.loc[gdp_df.groupby('ISO3')['cab'].filter(lambda x: len(x[pd.isnull(x)] ) < len(x)).index,:]

remaining_countries_list = gdp_df.ISO3.unique()

removed_countries_list = list(set(full_countries_list) - set(remaining_countries_list))
print('Filtered countries: '+ str(len(removed_countries_list)))
print('Remaining countries: '+str(len(remaining_countries_list)))

for iso in removed_countries_list:
    try:
        print(iso + ' | ' + pycountry.countries.get(alpha_3=iso).name)
    except:
        print(iso + ' | ' + '------------ NONE ------------')

gdp_df_copy = gdp_df.copy()

url = 'https://www.imf.org/external/pubs/ft/weo/data/WEOhistorical.xlsx'
weo_excel = pd.ExcelFile(url)
print(weo_excel.sheet_names)
df_weo = pd.read_excel(url, sheet_name="ngdp_rpch")
s_weo = df_weo[df_weo.columns[(~df_weo.columns.str.startswith('F'))]]
f_weo = df_weo[df_weo.columns[(~df_weo.columns.str.startswith('S'))]]
df_weo_melt = df_weo.melt(id_vars=['country','WEO_Country_Code','ISOAlpha_3Code','year'],
                          var_name="year_report",
                          value_name="gdp_pc_weo").dropna()
df_weo_melt = df_weo_melt[df_weo_melt.gdp_pc_weo!='.']
weo_forecast_1S = df_weo_melt[(df_weo_melt.year_report.str.startswith('S')) &
                              (df_weo_melt.year == df_weo_melt.year_report.str[1:5].astype('int') + 1)]
weo_forecast_1S.loc[:,'year_report'] = weo_forecast_1S.loc[:,'year'] - 1
weo_forecast_1S = weo_forecast_1S[(weo_forecast_1S.year_report >= start_year) & (weo_forecast_1S.year_report <= end_year)]
weo_forecast_1F = df_weo_melt[(df_weo_melt.year_report.str.startswith('F')) &
                              (df_weo_melt.year == df_weo_melt.year_report.str[1:5].astype('int') + 1)]
weo_forecast_1F.loc[:,'year_report'] = weo_forecast_1F.loc[:,'year'] - 1
weo_forecast_1F = weo_forecast_1F[(weo_forecast_1F.year_report >= start_year) & (weo_forecast_1F.year_report <= end_year)]

merge_fc_1S = pd.merge(gdp_df_copy, weo_forecast_1S, how='left', left_on=['ISO3','Year'], right_on=['ISOAlpha_3Code','year_report'])
merge_fc_1F = pd.merge(gdp_df_copy, weo_forecast_1F, how='left', left_on=['ISO3','Year'], right_on=['ISOAlpha_3Code','year_report'])

nan_weo_forecast = merge_fc_1S[merge_fc_1S.gdp_pc_weo.isna()].ISO3.unique()
nan_weo_forecast = np.unique(np.concatenate([nan_weo_forecast, merge_fc_1F[merge_fc_1F.gdp_pc_weo.isna()].ISO3.unique()]))

for iso in nan_weo_forecast:
    try:
        print(iso + ' | ' + pycountry.countries.get(alpha_3=iso).name)
    except:
        print(iso + ' | ' + '------------ NONE ------------')

for nan_country in nan_weo_forecast.tolist():
    gdp_df = gdp_df[gdp_df.ISO3 != nan_country]
    merge_fc_1S = merge_fc_1S[merge_fc_1S.ISO3 != nan_country]
    merge_fc_1F = merge_fc_1F[merge_fc_1F.ISO3 != nan_country]

try:
    gdp_graph_df = gdp_df.copy() # this will be used later for graphs
    gdp_df = gdp_df.drop(columns=['pop_growth_next'], inplace=False)
except:
    pass


cols_i = gdp_df.shape[1]
unemployment_col = gdp_df.unemployment
gdp_df_columns1 = gdp_df.columns
gdp_df = gdp_df.dropna(thresh=len(gdp_df)*0.7, axis='columns')
gdp_df.insert(10, 'unemployment', unemployment_col)
cols_f = gdp_df.shape[1]
print(gdp_df.columns)

list(set(gdp_df.columns.tolist())^set(gdp_df_columns1.tolist()))
print('Removed columns: '+str(cols_i - cols_f))

def mse_forecast_scaled(y_true, y_pred):
    scaler = StandardScaler()
    scaler.fit(np.asarray(y_true).reshape(-1,1))
    y_true = scaler.transform(np.asarray(y_true).reshape(-1,1))
    y_pred = scaler.transform(np.asarray(y_pred).reshape(-1,1))
    return mean_squared_error(y_true, y_pred)

def mae_forecast_scaled(y_true, y_pred):
    scaler = StandardScaler()
    scaler.fit(np.asarray(y_true).reshape(-1,1))
    y_true = scaler.transform(np.asarray(y_true).reshape(-1,1))
    y_pred = scaler.transform(np.asarray(y_pred).reshape(-1,1))
    return mean_absolute_error(y_true, y_pred)

print(mse_forecast_scaled(merge_fc_1F[merge_fc_1F.Year>=(start_year+1)].gdp_real_gwt_next, merge_fc_1F[merge_fc_1F.Year>=(start_year+1)].gdp_pc_weo))

print(mse_forecast_scaled(merge_fc_1S[merge_fc_1S.Year>=(start_year+1)].gdp_real_gwt_next, merge_fc_1S[merge_fc_1S.Year>=(start_year+1)].gdp_pc_weo))

gdp_graph_df['gdp_growth_converted'] = ((gdp_graph_df.gdp_real_gwt_next + 100)/100)
gdp_graph_df['pop_growth_converted'] = ((gdp_graph_df.pop_growth_next + 100)/100)
gdp_graph_df['gdp_pc_growth'] = gdp_graph_df['gdp_growth_converted'] / gdp_graph_df['pop_growth_converted']
gdp_growth_total = ((gdp_df.gdp_real_gwt + 100)/100)
gdp_growth_total = pd.DataFrame({'ISO3': gdp_df['ISO3'], 'gdp_growth_total': gdp_growth_total})
gdp_growth_total = gdp_growth_total.groupby('ISO3', as_index=False).prod().gdp_growth_total

pop_growth_total = ((gdp_df.pop_growth + 100)/100)
pop_growth_total = pd.DataFrame({'ISO3': gdp_df['ISO3'], 'pop_growth_total': pop_growth_total})
pop_growth_total = pop_growth_total.groupby('ISO3', as_index=False).prod().pop_growth_total

gdp_sns_plots = gdp_df.groupby('ISO3', as_index=False).last()
gdp_sns_plots['gdp_per_capita'] = gdp_sns_plots.gdp_real_us_fixed / gdp_sns_plots.population
gdp_sns_plots['log_gdp_per_capita'] = np.log(gdp_sns_plots.gdp_per_capita)
gdp_sns_plots['gdp_growth_total'] = gdp_growth_total
gdp_sns_plots['pop_growth_total'] = pop_growth_total
gdp_sns_plots['gdp_pc_growth_total'] = gdp_growth_total / pop_growth_total
gdp_sns_plots['log_co2_emissions'] = np.log(gdp_sns_plots['co2_emissions'])
print(gdp_sns_plots.columns)


url = 'https://databank.worldbank.org/data/download/site-content/CLASS.xlsx'
classifications = pd.ExcelFile(url)
classifications = pd.read_excel(url, sheet_name="List of economies", header= 0)
print(classifications)


classifications = classifications[['Code', 'Income group']]
classifications.columns = classifications.columns.str.replace('Code', 'ISO3')

gdp_sns_plots = gdp_sns_plots.merge(classifications[['ISO3','Income group']], how='left')


gdp_sns_plots = gdp_sns_plots.drop(['Year', 'gdp_real_gwt_next', 'gdp_real_gwt_prev', 'gdp_real_gwt',
       'service_value_added','gdp_per_capita'], axis=1)
print(gdp_sns_plots.columns)
print(gdp_sns_plots['Income group'].value_counts())


pd.set_option("display.max_rows", 20)
wb_out = gdp_df.copy()
wb_out = wb_out.sort_values(by=['Year','ISO3'],ascending=[True,True])

continents = []
for iso3 in wb_out.ISO3:
    try:
        continents = continents + [pycountry_convert.country_alpha2_to_continent_code(
            pycountry_convert.country_alpha3_to_country_alpha2(iso3))]
    except:
        continents = continents + ['OTHER']
wb_out.insert(0, 'continent_code', continents, 'continent_code')

wb_out = wb_out.merge(classifications[['ISO3','Income group']], how='left')
wb_out.columns = wb_out.columns.str.replace('Income group', 'income_group')
wb_out = wb_out[['income_group'] + [col for col in wb_out.columns if col != 'income_group']]
col_group = 'ISO3'
for df in [wb_out]:
    list_cols = df.isna().sum(0)[df.isna().sum(0)>0].index
    for colname in list_cols:
        df[colname] = df.groupby(col_group, sort=False)[colname].transform(lambda x: x.ffill().bfill())
col_group = ['income_group','continent_code','Year']
for df in [wb_out]:
    list_cols = df.isna().sum(0)[df.isna().sum(0)>0].index
    for colname in list_cols:
        df[colname] = df.groupby(col_group, sort=False)[colname].transform(lambda x: x.fillna(x.median()))
col_group = ['income_group','continent_code']
for df in [wb_out]:
    list_cols = df.isna().sum(0)[df.isna().sum(0)>0].index
    for colname in list_cols:
        df[colname] = df.groupby(col_group, sort=False)[colname].transform(lambda x: x.fillna(x.median()))
col_group = ['income_group']
for df in [wb_out]:
    list_cols = df.isna().sum(0)[df.isna().sum(0)>0].index
    for colname in list_cols:
        df[colname] = df.groupby(col_group, sort=False)[colname].transform(lambda x: x.fillna(x.median()))
index_target = wb_out.columns.get_loc("gdp_real_gwt_next")
# normalize data except for booleans:
for df in [wb_out]:
    df.iloc[:,index_target:] = StandardScaler().fit_transform(df.iloc[:,index_target:])

x_out = wb_out.iloc[:,(index_target):]
y_out = wb_out.iloc[:,index_target]

# One-class SVM

oc_svm = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.05)
oc_svm.fit(x_out)
pred = oc_svm.predict(x_out)
index_svm_out = np.where(pred==-1)
index_svm_out = index_svm_out[0].tolist()
values = x_out.iloc[index_svm_out,:]

plt.rcParams["figure.figsize"] = (5, 5)
plt.scatter(x_out.loc[:,'life_expectancy'], x_out.loc[:,'agff_gdp'])
plt.scatter(values.loc[:,'life_expectancy'], values.loc[:,'agff_gdp'], color='tab:red')
plt.show()

# fit the model for outlier detection (default)
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred = clf.fit_predict(x_out)
X_scores = clf.negative_outlier_factor_
sns.set_style('whitegrid')
sns.kdeplot(X_scores, bw_method=0.5)

x_axis_col = 'life_expectancy'
y_axis_col = 'cab'
plt.rcParams["figure.figsize"] = (8, 5)

plt.scatter(x_out.loc[:,x_axis_col], x_out.loc[:,y_axis_col], color="k", s=3.0, label="Data points")

radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
plt.scatter(
    x_out.loc[:,x_axis_col],
    x_out.loc[:,y_axis_col],
    s=1000 * radius,
    edgecolors="tab:red",
    facecolors="none",
    label="Outlier scores",
)
plt.axis("tight")
legend = plt.legend(loc="upper left")
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()