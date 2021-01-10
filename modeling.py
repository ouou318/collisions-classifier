# Import
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from functools import reduce

# ML
from sklearn.linear_model import SGDClassifier

# Ensemble method
from sklearn.ensemble import RandomForestClassifier

# Split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Metric to evaluation
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import recall_score, pairwise_distances
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc

# Standard scaler
from sklearn.preprocessing import StandardScaler

# Pipeline
from sklearn.pipeline import make_pipeline

# Utils
from sklearn.utils import resample

# Import Data
df_join = pd.read_csv("data/df_join.csv")
df_join = df_join.drop(columns='Unnamed: 0', axis=1)
# Notice that we will have to deal with imbalanced data
# 1     137026
# 2      58472
# 2b      3082
# 3        346

# Create X and y
y = df_join['SEVERITYCODE'].astype('object')
y = y.to_frame()
X = df_join.drop('SEVERITYCODE', axis=1)

# # Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
# # Resampling
# df_train = pd.merge(X_train, y_train, left_index=True, right_index=True)
# df_majority = df_train[df_train.SEVERITYCODE == "1"]
# df_2 = df_train[df_train.SEVERITYCODE == "2"]
# df_2b = df_train[df_train.SEVERITYCODE == "2b"]
# df_3 = df_train[df_train.SEVERITYCODE == "3"]
#
# # Upsample minority class
# df_2_upsampled = resample(df_2, replace=True, n_samples=91350, random_state=123)
# df_2b_upsampled = resample(df_2b, replace=True, n_samples=91350, random_state=123)
# df_3_upsampled = resample(df_3, replace=True, n_samples=91350, random_state=123)
#
# # Downsample minority class
# df_majority_downsampled = resample(df_majority, replace=True, n_samples=231, random_state=123)
# df_2_downsampled = resample(df_2, replace=True, n_samples=231, random_state=123)
# df_2b_downsampled = resample(df_2b, replace=True, n_samples=231, random_state=123)
#
# # Combine majority class with upsampled minority class
# df_upsampled = pd.concat([df_majority, df_2_upsampled, df_2b_upsampled, df_3_upsampled])
# df_downsampled = pd.concat([df_majority_downsampled, df_2_downsampled, df_2b_downsampled, df_3])
#
# # Display new class counts
# df_upsampled.SEVERITYCODE.value_counts()
# df_downsampled.SEVERITYCODE.value_counts()
#
# # Upsample X_train and y_train
# y_upsample = df_upsampled['SEVERITYCODE'].astype('object')
# X_upsample = df_upsampled.drop('SEVERITYCODE', axis=1)
#
# X_upsample = X_upsample.reset_index()
# X_upsample = X_upsample.drop(columns='index', axis=1)
# y_upsample = y_upsample.reset_index()
# y_upsample = y_upsample.drop(columns='index', axis=1)
#
# # Down sample X_train and y_train
# y_downsample = df_downsampled['SEVERITYCODE'].astype('object')
# X_downsample = df_downsampled.drop('SEVERITYCODE', axis=1)
#
# X_downsample = X_downsample.reset_index()
# X_downsample = X_downsample.drop(columns='index', axis=1)
# y_downsample = y_downsample.reset_index()
# y_downsample = y_downsample.drop(columns='index', axis=1)


# # Baseline Model
# forest_clf1 = RandomForestClassifier(
#     random_state=42
# )
# forest_clf1.fit(X_train, y_train.values.ravel())
# y_pred = forest_clf1.predict(X_test)
# print('Baseline model')
# print(recall_score(y_test, y_pred, average=None))
# print(precision_score(y_test, y_pred, average=None))
# print(f1_score(y_test, y_pred, average=None))
# # [0.90471263 0.37312963 0.         0.        ]
# # [0.76418393 0.59537572 0.         0.        ]
# # [0.8285317 0.4587529 0.        0.       ]
#
# # Try SGD Classifier
# sgd_clf = make_pipeline(
#     StandardScaler(),
#     SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
# )
# sgd_clf.fit(X_upsample, y_upsample.values.ravel())
# y_pred = sgd_clf.predict(X_test)
# print('SGD model')
# print(precision_score(y_test, y_pred, average=None))
# print(recall_score(y_test, y_pred, average=None))
# print(f1_score(y_test, y_pred, average=None))
# # [0.86387363 0.4041865  0.04068667 0.01102656]
# # [0.5658756  0.45923243 0.30465587 0.65740741]
# # [0.68381923 0.42995478 0.07178631 0.02168932]
#
# # Upsampling
# forest_clf = RandomForestClassifier(
#     random_state=42,
#     criterion="entropy",
#     # max_depth=10,
#     min_samples_split=2,
#     n_estimators=200,
#     max_features="sqrt",
#     class_weight='balanced',
#     n_jobs=-1
# )
# forest_clf.fit(X_upsample, y_upsample.values.ravel())
# y_pred = forest_clf.predict(X_test)
# print('Upsampling model')
# print(recall_score(y_test, y_pred, average=None))
# print(precision_score(y_test, y_pred, average=None))
# print(f1_score(y_test, y_pred, average=None))
# # [0.87151052 0.41906862 0.00303644 0.        ]
# # [0.77164994 0.5597083  0.02608696 0.        ]
# # [0.81854579 0.47928426 0.00543971 0.        ]
#
# # Downsampling
# forest_clf = RandomForestClassifier(
#     random_state=42,
#     criterion="entropy",
#     # max_depth=10,
#     min_samples_split=2,
#     n_estimators=200,
#     max_features="sqrt",
#     class_weight='balanced',
#     n_jobs=-1
# )
# forest_clf.fit(X_downsample, y_downsample.values.ravel())
# y_pred = forest_clf.predict(X_test)
# print('Downsampling model')
# print(recall_score(y_test, y_pred, average=None))
# print(precision_score(y_test, y_pred, average=None))
# print(f1_score(y_test, y_pred, average=None))
# # [0.5658756  0.45923243 0.30465587 0.65740741]
# # [0.86387363 0.4041865  0.04068667 0.01102656]
# # [0.68381923 0.42995478 0.07178631 0.02168932]
#
#
# # Redesign the question: Try binary model
# # Calculate baseline
# features_add = [
#     'temp', 'dewp', 'slp', 'stp', 'visib', 'wdsp',
#     'mxpsd', 'gust', 'max', 'min', 'prcp', 'sndp', 'fog', 'rain_drizzle',
#     'snow_ice_pellets', 'hail', 'thunder', 'tornado_funnel_cloud',
#     'count', 'dangerous', 'safe', 'na_num', 'Xcount'
# ]
# df_orig = df_join.drop(features_add, axis=1)
# y_or = df_orig['SEVERITYCODE'].astype('object')
# y_or = y_or.to_frame()
# y_or.loc[y_or['SEVERITYCODE'] != "3", :] = "1"
# X_or = df_orig.drop('SEVERITYCODE', axis=1)
#
# X_train_or, X_test_or, y_train_or, y_test_or = train_test_split(
#     X_or, y_or, test_size=0.33, random_state=42
# )
# forest_clf1 = RandomForestClassifier(
#     random_state=42
# )
# model1 = forest_clf1.fit(X_train_or, y_train_or.values.ravel())
# y_pred = forest_clf1.predict(X_test_or)
# print(recall_score(y_test_or, y_pred, average=None))
# print(precision_score(y_test_or, y_pred, average=None))
# print(f1_score(y_test_or, y_pred, average=None))
# # [0.99998449 0.        ]
# # [0.9983281 0.       ]
# # [0.99915561 0.        ]
#
# # ROC
# probs = model1.predict_proba(X_test_or)
# preds = probs[:, 1]
# fpr, tpr, threshold = roc_curve(y_test_or, preds, pos_label="3")
# roc_auc_baseline = auc(fpr, tpr)
# # 0.7208326872383315

# Try adjusted model
y_bi = y.copy()
y_bi.loc[y_bi['SEVERITYCODE'] != "3", :] = "1"

# Split
X_train_bi, X_test_bi, y_train_bi, y_test_bi = train_test_split(
    X, y_bi, test_size=0.33, random_state=42
)

# Resampling
df_train_bi = pd.merge(X_train_bi, y_train_bi, left_index=True, right_index=True)
df_majority_bi = df_train_bi[df_train_bi.SEVERITYCODE == "1"]
df_3_bi = df_train_bi[df_train_bi.SEVERITYCODE == "3"]

df_majority_dnsampled = resample(df_majority_bi, replace=True, n_samples=231, random_state=123)
df_dnsampled = pd.concat([df_majority_dnsampled, df_3_bi])

y_dnsample = df_dnsampled['SEVERITYCODE'].astype('object')
X_dnsample = df_dnsampled.drop('SEVERITYCODE', axis=1)

X_dnsample = X_dnsample.reset_index()
X_dnsample = X_dnsample.drop(columns='index', axis=1)
y_dnsample = y_dnsample.reset_index()
y_dnsample = y_dnsample.drop(columns='index', axis=1)

# Run forest clf
forest_clf = RandomForestClassifier(
    random_state=42,
    criterion="entropy",
    # max_depth=10,
    min_samples_split=2,
    n_estimators=200,
    max_features="sqrt",
    class_weight='balanced',
    n_jobs=-1
)

model = forest_clf.fit(X_dnsample, y_dnsample.values.ravel())
y_pred = forest_clf.predict(X_test_bi)
print(recall_score(y_test_bi, y_pred, average=None))
print(precision_score(y_test_bi, y_pred, average=None))
print(f1_score(y_test_bi, y_pred, average=None))
# [0.8135835  0.91666667]
# [0.9998285  0.00816764]
# [0.89714193 0.01619102]

# ROC
model = forest_clf.fit(X_dnsample, y_dnsample.values.ravel())
probs = model.predict_proba(X_test_bi)
preds = probs[:, 1]
fpr, tpr, threshold = roc_curve(y_test_bi, preds, pos_label="3")
roc_auc_baseline = auc(fpr, tpr)
print(roc_auc_baseline)
# 0.9273012468197768

# # Cross-validation
# # Use KFold
# kf = KFold(n_splits=10, random_state=42, shuffle=True)
# forest_clf = RandomForestClassifier(
#     random_state=42,
#     criterion="entropy",
#     # max_depth=10,
#     min_samples_split=2,
#     n_estimators=200,
#     max_features="sqrt",
#     class_weight="balanced",
#     n_jobs=-1
# )
# for train_index, test_index in kf.split(X_dnsample):
#     X_train, X_test = X_dnsample.loc[train_index, :], X_dnsample.loc[test_index, :]
#     y_train, y_test = y_dnsample.loc[train_index, :], y_dnsample.loc[test_index, :]
#     model = forest_clf.fit(X_train, y_train.values.ravel())
#     probs = model.predict_proba(X_test)
#     preds = probs[:, 1]
#     fpr, tpr, threshold = roc_curve(y_test, preds, pos_label="3")
#     roc_auc_baseline = auc(fpr, tpr)
#     print(roc_auc_baseline)

# 0.9574074074074074
# 0.9601449275362318
# 0.9290060851926978
# 0.8935483870967743
# 0.9220272904483431
# 0.9404761904761905
# 0.9356060606060607
# 0.8513257575757576
# 0.9365384615384615
# 0.8109640831758034

# Feature importance
baseline = roc_auc_baseline

# Permutation importances
forest_clf = RandomForestClassifier(
    random_state=42,
    criterion="entropy",
    # max_depth=10,
    min_samples_split=2,
    n_estimators=200,
    max_features="sqrt",
    class_weight='balanced',
    n_jobs=-1
)


def permutation_importances(rf, X_train, y_train):
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        np.random.seed(42)
        X_train.loc[:, col] = np.random.permutation(X_train[col])
        model = forest_clf.fit(X_train, y_train.values.ravel())
        probs = model.predict_proba(X_test_bi)
        preds = probs[:, 1]
        fpr, tpr, threshold = roc_curve(y_test_bi, preds, pos_label="3")
        roc_auc = auc(fpr, tpr)
        X_train[col] = save
        imp.append(roc_auc)
    return imp


perm_output = permutation_importances(forest_clf, X_dnsample, y_dnsample)
result = pd.DataFrame(perm_output)

perm_test = baseline - result
perm_test.columns = ['diff']
perm_test['col'] = X.columns


# Drop columns importances
forest_clf = RandomForestClassifier(
    random_state=42,
    criterion="entropy",
    # max_depth=10,
    min_samples_split=2,
    n_estimators=200,
    max_features="sqrt",
    class_weight='balanced',
    n_jobs=-1
)


def dropcol_importances(rf, X_train, y_train):
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        X_test_dp = X_test_bi.drop(col, axis=1)
        model = forest_clf.fit(X, y_train.values.ravel())
        probs = model.predict_proba(X_test_dp)
        preds = probs[:, 1]
        fpr, tpr, threshold = roc_curve(y_test_bi, preds, pos_label="3")
        roc_auc = auc(fpr, tpr)
        imp.append(roc_auc)
    imp = np.array(imp)
    return imp


drop_output = dropcol_importances(forest_clf, X_dnsample, y_dnsample)
drop_result = pd.DataFrame(drop_output)

drop_test = baseline - drop_result
drop_test.columns = ['diff']
drop_test['col'] = X.columns

import pdb
pdb.set_trace()

perm_test.to_csv('data/perm_rank.csv')
drop_test.to_csv('data/drop_rank.csv')

drop_test.sort_values(by='diff', ascending=False).head(5)
perm_test.sort_values(by='diff', ascending=False).head(5)

drop_test.sort_values(by='diff').head(5)
perm_test.sort_values(by='diff').head(5)


# Feature important from random forest
forest_clf = RandomForestClassifier(
    random_state=42,
    criterion="entropy",
    # max_depth=10,
    min_samples_split=2,
    n_estimators=200,
    max_features="sqrt",
    class_weight='balanced',
    n_jobs=-1
)
model = forest_clf.fit(X_dnsample, y_dnsample.values.ravel())
importance = forest_clf.feature_importances_
features = pd.DataFrame(importance, index=X.columns)
features.columns = ['value']
features.sort_values(by="value", ascending=False)
