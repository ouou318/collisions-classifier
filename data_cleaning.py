# Import
import pandas as pd
from functools import reduce
from sklearn.preprocessing import OneHotEncoder

# Seattle Collision Data
# Data dictionary https://www.seattle.gov/Documents/Departments/SDOT/GIS/Collisions_OD.pdf
X_org = pd.read_csv("data/Collisions.csv")
str_org = pd.read_csv("data/Seattle_Streets.csv")
weather = pd.read_csv("data/weather_seattle.csv")

# Remove severitycode na or unknown
X_org = X_org.loc[~(X_org['SEVERITYCODE'].isna())].copy().reset_index()
X_org = X_org.loc[~(X_org['SEVERITYDESC'] == "Unknown")].copy().reset_index()

# Feature Engineering
# Count number of accidents
count = X_org[~(X_org['X'].isna())].groupby(['X', 'Y']).size().reset_index()
count = count.rename({0: 'count'}, axis='columns').copy()
X_org = pd.merge(X_org, count, how='left', on=['X', 'Y'])

# Create dangerous and safe variables
X_org['dangerous'] = 0
X_org.loc[X_org['SPEEDING'] == "Y", 'dangerous'] = X_org.loc[
    X_org['SPEEDING'] == "Y", 'dangerous'] + 1
X_org.loc[X_org['INATTENTIONIND'] == "Y", 'dangerous'] = X_org.loc[
    X_org['INATTENTIONIND'] == "Y", 'dangerous'] + 1
X_org.loc[
    X_org['ST_COLDESC'] == "Vehicle going straight hits pedestrian", 'dangerous'
] = X_org.loc[X_org['ST_COLDESC'] == "Vehicle going straight hits pedestrian", 'dangerous'] + 1
X_org.loc[
    X_org['COLLISIONTYPE'] == "Pedestrian", 'dangerous'
] = X_org.loc[X_org['COLLISIONTYPE'] == "Pedestrian", 'dangerous'] + 1
X_org['safe'] = 0
X_org.loc[
    X_org['ST_COLDESC'] == "One parked--one moving", 'safe'
] = X_org.loc[X_org['ST_COLDESC'] == "One parked--one moving", 'safe'] + 1
X_org.loc[
    X_org['COLLISIONTYPE'] == "Parked Car", 'safe'
] = X_org.loc[X_org['COLLISIONTYPE'] == "Parked Car", 'safe'] + 1

# Select useful variable
X_org["INCDTTM_copy"] = X_org["INCDTTM"]

X = X_org[[
   'X', 'Y', 'INCDTTM_copy', 'ADDRTYPE', 'SEVERITYCODE', 'COLLISIONTYPE',
   'PERSONCOUNT', 'PEDCOUNT', 'PEDCYLCOUNT', 'VEHCOUNT', 'INJURIES',
   'SERIOUSINJURIES', 'FATALITIES', 'INCDTTM', 'JUNCTIONTYPE',
   'SDOT_COLCODE', 'SDOT_COLDESC', 'INATTENTIONIND', 'UNDERINFL',
   'WEATHER', 'ROADCOND', 'LIGHTCOND', 'PEDROWNOTGRNT', 'SDOTCOLNUM',
   'SPEEDING', 'ST_COLCODE', 'ST_COLDESC', 'SEGLANEKEY', 'CROSSWALKKEY',
   'HITPARKEDCAR', 'count', 'dangerous', 'safe'
]].copy()

# Extract time from datetime variable
X.INCDTTM = pd.to_datetime(X.INCDTTM).copy()
X.loc[:, 'year'] = X.INCDTTM.dt.year
X.loc[:, 'mo'] = X.INCDTTM.dt.month
X.loc[:, 'da'] = X.INCDTTM.dt.day
X.loc[:, 'hour'] = X.INCDTTM.dt.hour
drop = [
    'INCDTTM',  # remove after extract information
    'INJURIES', 'SERIOUSINJURIES', 'FATALITIES',  # accident related variables
    'SDOT_COLDESC', 'ST_COLDESC',  # repeated columns with description and code
    'SDOT_COLCODE', 'ST_COLCODE', 'SDOTCOLNUM', 'SEGLANEKEY', 'CROSSWALKKEY'  # remove id number
]
X = X.drop(drop, axis=1).copy()

# Cleaning
# Looking at missing value percentage
missing_df = X.isna().sum().to_frame().reset_index().rename(
    columns={'index': 'variables', 0: 'count'}
)
missing_df['perc'] = missing_df['count']/198926
drop_missing = missing_df['variables'][missing_df['perc'] > 0.95].to_list()
# X = X.drop(drop_missing, axis=1).copy()

# Looking at correlation
corr_tab = X.select_dtypes(
    include=['int64', 'float64']
).corr().abs().unstack().to_frame().reset_index().rename(columns={0: "correlation"})
corr_tab[
    (corr_tab['correlation'] > 0.9) & (corr_tab['correlation'] != 1)
].sort_values(by='correlation', ascending=False)  # no variables need to be dropped
# X = X.drop(drop_corr, axis=1).copy()

# Checking Nan
na_df = pd.DataFrame(X.isna().sum())
na_list = na_df[na_df[0] != 0].index
print(na_list)
"""
'ADDRTYPE', 'COLLISIONTYPE', 'JUNCTIONTYPE', 'INATTENTIONIND', 'UNDERINFL', 'WEATHER',
'ROADCOND', 'LIGHTCOND', 'ST_COLCODE', 'count'
"""

# Drop observations that contains little information
X['na_num'] = X.isnull().sum(axis=1)  # X = X[X['na_num'].isin([7, 8, 9])].copy()

"""
We have some binary variables 'INATTENTIONIND', 'UNDERINFL', 'HITPARKEDCAR', 'SPEEDING',
'PEDROWNOTGRNT'.
We want to clean up to imputation or create missing category
"""
binary = ['INATTENTIONIND', 'UNDERINFL', 'HITPARKEDCAR', 'SPEEDING', 'PEDROWNOTGRNT']
X.loc[:, binary] = X.loc[:, binary].fillna('MISSING')
for i in X[binary]:
    print(X[i].value_counts())
# UNDERINFL
X.loc[X['UNDERINFL'] == "Y", 'UNDERINFL'] = "1"
X.loc[X['UNDERINFL'] == "N", 'UNDERINFL'] = "0"
X.loc[X['UNDERINFL'] == "MISSING", 'UNDERINFL'] = "0"
# PEDROWNOTGRNT
X.loc[X['PEDROWNOTGRNT'] == "MISSING", 'PEDROWNOTGRNT'] = "0"
# SPEEDING
X.loc[X['SPEEDING'] == "MISSING", 'SPEEDING'] = "0"
# HITPARKEDCAR
X.loc[X['HITPARKEDCAR'] == "MISSING", 'HITPARKEDCAR'] = "0"
# INATTENTIONIND
X.loc[X['INATTENTIONIND'] == "MISSING", 'INATTENTIONIND'] = "0"
for i in binary:
    X.loc[X[i] == "Y", i] = "1"
    X.loc[X[i] == "N", i] = "0"

# Filling in Nan
mask_obj = X.select_dtypes(include='object').columns
mask_num = X.select_dtypes(include=['float64', 'int64']).columns
X.loc[:, mask_obj] = X.loc[:, X.select_dtypes(include='object').columns].fillna('MISSING')
X['Xcount'] = 0
X.loc[X['count'].isna(), 'Xcount'] = 1
X.loc[:, mask_num] = X.loc[:, X.select_dtypes(include=['float64', 'int64']).columns].fillna(0)

# Identify numberic columns
num_col = [
    "PERSONCOUNT", "PEDCOUNT", "PEDCYLCOUNT", "VEHCOUNT", 'year', 'mo', 'hour', 'da',
    'count', 'na_num', 'dangerous', 'safe', 'Xcount', 'INATTENTIONIND', 'UNDERINFL',
    'HITPARKEDCAR',
    'SPEEDING', 'PEDROWNOTGRNT'
]
num_mask = X.columns.isin(num_col)
cat_col = X.columns[~num_mask].tolist()
cat_col = ['ADDRTYPE', 'COLLISIONTYPE', 'JUNCTIONTYPE', 'WEATHER', 'ROADCOND', 'LIGHTCOND']

# Fill missing values with 0
X.loc[:, num_col] = X.loc[:, num_col].apply(lambda x: x.astype(int))
X.loc[:, cat_col] = X.loc[:, cat_col].apply(lambda x: x.astype(object))
# See if value makes sense
# for i in  X.loc[:, cat_col].columns: X[i].value_counts()
# for i in  X.loc[:, num_col].columns: X[i].value_counts()

# Create LabelEncoder object: le
df = X.drop(cat_col, axis=1).copy()
enc = OneHotEncoder(handle_unknown='ignore')
for i in cat_col:
    enc_df = pd.DataFrame(enc.fit_transform(X[[i]]).toarray())
    enc_df = enc_df.add_prefix(i)
    # merge with main df on key values
    df = df.join(enc_df)

# Add weather data
df_935_rename = weather[weather.stn == 727935].drop(
    ["stn", "wban", "count_temp", "count_dewp", "count_slp", "count_stp", "count_visib",
     "count_wdsp", "flag_max", "flag_min", "flag_prcp", "usaf", "wban_1", "name", "country",
     "state", "call", "lat", "lon", "elev", "begin", "end"], axis=1
)
# vars_list9999 = [
#     'temp', 'dewp', 'slp', 'stp',
#     'max', 'min'
# ]
# vars_list999 = [
#     'gust', 'visib', 'wdsp', 'mxpsd', 'sndp'
# ]
# vars_list99 = ['prcp']
# for i in vars_list9999:
#     df_935_rename.loc[df_935_rename[i] == 9999.9, i] = np.NaN
# for i in vars_list999:
#     df_935_rename.loc[df_935_rename[i] == 999.9, i] = np.NaN
# for i in vars_list99:
#     df_935_rename.loc[df_935_rename[i] == 99.9, i] = np.NaN

dewp_median = df_935_rename.groupby(['year', 'mo'])['dewp'].median().reset_index()
slp_median = df_935_rename.groupby(['year', 'mo'])['slp'].median().reset_index()
stp_median = df_935_rename.groupby(['year', 'mo'])['stp'].median().reset_index()
max_median = df_935_rename.groupby(['year', 'mo'])['max'].median().reset_index()
data_frames = [dewp_median, slp_median, stp_median, max_median]
df_merged = reduce(
   lambda left, right: pd.merge(left, right, on=['year', 'mo'], how='left'), data_frames
)
df_935 = pd.merge(df_935_rename, df_merged, how="left", on=['year', 'mo'])

df_935.loc[df_935['dewp_x'] == 9999.9, 'dewp_x'] = df_935['dewp_y']
df_935.loc[df_935['slp_x'] == 9999.9, 'slp_x'] = df_935['slp_y']
df_935.loc[df_935['stp_x'] == 9999.9, 'stp_x'] = df_935['stp_y']
df_935.loc[df_935['max_x'] == 9999.9, 'max_x'] = df_935['max_y']
df_935.loc[df_935['stp_x'] == 9999.9, 'stp_x'] = 0
df_935.loc[df_935['prcp'] == 99.99, 'prcp'] = 0
df_935.loc[df_935['gust'] == 999.9, 'gust'] = 0
df_935.loc[df_935['sndp'] == 999.9, 'sndp'] = 0
df_935_final = df_935.drop(['dewp_y', 'slp_y', 'stp_y', 'max_y'], axis=1)

df_join = pd.merge(
    df_935_rename,
    df,
    how='right',
    left_on=['year', 'mo', 'da'],
    right_on=['year', 'mo', 'da']
)
df_join = df_join[~df_join.temp.isna()]

# Looking at correlation
corr_tab = df_join.select_dtypes(
    include=['int64', 'float64']
).corr().abs().unstack().to_frame().reset_index().rename(columns={0: "correlation"})
corr_tab[
    (corr_tab['correlation'] > 0.9) & (corr_tab['correlation'] != 1)
].sort_values(by='correlation', ascending=False)  # no variables need to be dropped
# X = X.drop(drop_corr, axis=1).copy()

# Export data
df_join.to_csv("data/df_latlon.csv")
df_join = df_join.drop(['X', 'Y', 'INCDTTM_copy'], axis=1)
df_join.to_csv("data/df_join.csv")
