import os
import pdb
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Encode
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# ML
from sklearn.tree import DecisionTreeClassifier

# Ensemble method
from sklearn.ensemble import RandomForestClassifier

# Split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

# Metric to evaluation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Standard scaler
from sklearn.preprocessing import StandardScaler

# Pipeline
from sklearn.pipeline import make_pipeline

# What causes a more severe accident?
# Data dictionary https://www.seattle.gov/Documents/Departments/SDOT/GIS/Collisions_OD.pdf
X_org = pd.read_csv("/Users/ou/Projects/traffic_collisions_ml_team2/data/Collisions.csv")
str_org = pd.read_csv("/Users/ou/Projects/traffic_collisions_ml_team2/data/Seattle_Streets.csv")

#test = pd.merge(X_org, str_org, left_on='LOCATION', right_on='UNITDESC', how='left')
#test[['LOCATION', 'UNITDESC']]

def clean(X_org):

    # Drop 'LOCATION'
    # Drop 'REPORTNO', 'STATUS', 'EXCEPTRSNCODE', 'EXCEPTRSNDESC', 'ST_COLCODE'
    # Drop 'INJURIES', 'SERIOUSINJURIES', 'FATALITIES', 'SEVERITYDESC, 'SDOT_COLDESC'
    # Convert INCDATE, INCDTTM (the same, thus only keep one)
    X_org = X_org.loc[~(X_org['SEVERITYCODE'].isna())].copy().reset_index()
    X_org = X_org.loc[~(X_org['SEVERITYDESC']=="Unknown")].copy().reset_index()

    count = X_org[~(X_org['X'].isna())].groupby(['X','Y']).size().reset_index()
    count = count.rename({0: 'count'}, axis='columns').copy()
    X_org = pd.merge(X_org, count, how='left', on=['X', 'Y'])


    X_org['dangerous'] = 0
    X_org.loc[X_org['SPEEDING']=="Y", 'dangerous'] = X_org.loc[X_org['SPEEDING']=="Y", 'dangerous'] + 1
    X_org.loc[X_org['INATTENTIONIND']=="Y", 'dangerous'] = X_org.loc[X_org['INATTENTIONIND']=="Y", 'dangerous'] + 1
    X_org.loc[X_org['ST_COLDESC']=="Vehicle going straight hits pedestrian", 'dangerous'] = X_org.loc[X_org['ST_COLDESC']=="Vehicle going straight hits pedestrian", 'dangerous'] + 1
    X_org.loc[X_org['COLLISIONTYPE']=="Pedestrian", 'dangerous'] = X_org.loc[X_org['COLLISIONTYPE']=="Pedestrian", 'dangerous'] + 1

    X_org['safe'] = 0
    X_org.loc[X_org['ST_COLDESC']=="One parked--one moving", 'safe'] = X_org.loc[X_org['ST_COLDESC']=="One parked--one moving", 'safe'] + 1
    X_org.loc[X_org['COLLISIONTYPE']=="Parked Car", 'safe'] = X_org.loc[X_org['COLLISIONTYPE']=="Parked Car", 'safe'] + 1


    X = X_org[[
        'SEVERITYDESC', 'ADDRTYPE', 'COLLISIONTYPE', 'PERSONCOUNT',
        'PEDCOUNT', 'PEDCYLCOUNT', 'VEHCOUNT', 'INJURIES', 'SERIOUSINJURIES',
        'FATALITIES', 'INCDTTM', 'JUNCTIONTYPE',
        'SDOT_COLDESC', 'INATTENTIONIND', 'UNDERINFL', 'WEATHER', 'ROADCOND',
        'LIGHTCOND', 'PEDROWNOTGRNT', 'SDOT_COLCODE', 'SDOTCOLNUM', 'SPEEDING', 'ST_COLCODE',
        'ST_COLDESC', 'SEGLANEKEY', 'CROSSWALKKEY', 'HITPARKEDCAR', 'count'
    ]].copy()
    y = X_org['SEVERITYCODE'].astype('object')

    X.INCDTTM = pd.to_datetime(X.INCDTTM).copy()
    X.loc[:, 'year'] = X.INCDTTM.dt.year
    X.loc[:, 'month'] = X.INCDTTM.dt.month
    X.loc[:, 'hour'] = X.INCDTTM.dt.hour

    drop = ['INCDTTM', 'INJURIES', 'SERIOUSINJURIES', 'FATALITIES', 'SEVERITYDESC',
            'SDOT_COLDESC', 'ST_COLDESC',  # repeated columns with description and code
            'SDOT_COLCODE', 'SDOTCOLNUM', 'SEGLANEKEY', 'CROSSWALKKEY']
    # ID for accident, lane segment, and crosswalk
    X = X.drop(drop, axis=1).copy()

    # Cleaning
    X.loc[X['UNDERINFL']=="Y", 'UNDERINFL'] = "1"
    X.loc[X['UNDERINFL']=="N", 'UNDERINFL'] = "0"

    X.loc[:, X.select_dtypes(include='object').columns] = X.loc[:, X.select_dtypes(include='object').columns].fillna('MISSING')
    X.loc[:, X.select_dtypes(include=['float64', 'int64']).columns] = X.loc[:, X.select_dtypes(include=['float64', 'int64']).columns].fillna(0)
    # only one observation has SDOT_COLCODE as NaN

    num_col = ["PERSONCOUNT", "PEDCOUNT", "PEDCYLCOUNT", "VEHCOUNT", 'year', 'month', 'hour',
        'INATTENTIONIND', 'UNDERINFL', 'PEDROWNOTGRNT', 'SPEEDING', 'HITPARKEDCAR', 'count'
    ]
    num_mask = X.columns.isin(num_col)
    cat_col = X.columns[~num_mask].tolist()

    # 'SPEEDING', 'PEDROWNOTGRNT', 'INATTENTIONIND' only has Y and MISSING
    binary = ['INATTENTIONIND', 'UNDERINFL', 'PEDROWNOTGRNT', 'SPEEDING', 'HITPARKEDCAR']
    for i in binary:
        X.loc[X[i]=="Y", i] = "1"
        X.loc[X[i]=="N", i] = "0"
        X.loc[X[i]=="MISSING", i] = "2"

    # Fill missing values with 0
    X.loc[:, num_col] = X.loc[:, num_col].apply(lambda x: x.astype(int))
    X.loc[:, cat_col] = X.loc[:, cat_col].apply(lambda x: x.fillna('MISSING'))
    binary_v2 = ['UNDERINFL']
    X.loc[:, binary_v2] = X.loc[:, binary_v2].apply(lambda x: x.astype(object))

    # Experiment
    # drop = ['year', 'month', 'hour']
    # X = X.drop(drop, axis=1).copy()
    # See if value makes sense
    # for i in  X.loc[:, cat_col].columns: X[i].value_counts()
    # for i in  X.loc[:, num_col].columns: X[i].value_counts()

    # I dont know what is up, but just leave it as is.
    # X.loc[X['ST_COLCODE']==' ', 'ST_COLCODE'] =

    # Create LabelEncoder object: le
    df = X.select_dtypes(include=['float64', 'int64'])
    enc = OneHotEncoder(handle_unknown='ignore')
    for i in X.select_dtypes(include='object'):
        enc_df = pd.DataFrame(enc.fit_transform(X[[i]]).toarray())
        enc_df = enc_df.add_prefix(i)

        # merge with main df on key values
        df = df.join(enc_df)

    drop = ['UNDERINFL2']
    df = df.drop(drop, axis=1).copy()

    return df, X, y


df, X, y = clean(X_org)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)


# Check to see if there is NaN
X_train.isna().sum().sort_values()
y_train.isna().sum()
X_test.isna().sum().sort_values()
df.isna().sum().sort_values()
y.isna().sum()


pdb.set_trace()
"""
k-Nearest Neighbors.
Decision Trees.
Naive Bayes.
Random Forest.
Gradient Boosting.
"""

# Ridge Classifier
from sklearn.linear_model import RidgeClassifier
clf_rg = RidgeClassifier()
# clf_rg.fit(X_train, y_train.values.ravel())
# y_pred = clf_rg.predict(X_test)
# print(precision_score(y_test, y_pred, average=None))
# print(recall_score(y_test, y_pred, average=None))
# print(accuracy_score(y_test, y_pred))
# print(f1_score(y_test, y_pred, average=None))

# Random forest classifier;
# This time Scikit-Learn did not have to run OvA or OvO
# because Random Forest classifiers can directly classify instances into multiple classes.
# forest_clf = RandomForestClassifier(random_state=42)
# The former is the number of trees in the forest. The larger the better, but also the longer
# it will take to compute. In addition, note that results will stop getting significantly better
# beyond a critical number of trees. The latter is the size of the random subsets of features to
# consider when splitting a node. The lower the greater the reduction of variance, but also the
# greater the increase in bias.
forest_clf = RandomForestClassifier(
    random_state=42, max_depth=None,
    min_samples_split=2,
    n_estimators=200, max_features="sqrt"
)
# forest_clf.fit(X_train, y_train.values.ravel())
# y_pred = forest_clf.predict(X_test)
# print(precision_score(y_test, y_pred, average=None))
# print(recall_score(y_test, y_pred, average=None))
# print(accuracy_score(y_test, y_pred))
# print(f1_score(y_test, y_pred, average=None))
# [0.7742471  0.57222338 0.11666667 0.        ]
# [0.87892426 0.42447379 0.00669856 0.        ]
# 0.7293361362459251
# [0.82327165 0.48739745 0.01266968 0.        ]

# # K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
# Kneigb_clf = NearestCentroid()

Kneigb_clf = KNeighborsClassifier(weights="distance", algorithm="auto")
y_pred = Kneigb_clf.predict(X_test)
print(precision_score(nca.transform(y_test), y_pred, average=None))
print(recall_score(nca.transform(y_test), y_pred, average=None))
print(accuracy_score(nca.transform(y_test), y_pred))
print(f1_score(nca.transform(y_test), y_pred, average=None))
# [0.73193164 0.47964778 0.         0.        ]
# [0.86983416 0.29787454 0.         0.        ]
# 0.6856015598817902
# [0.79494661 0.36751321 0.         0.        ]

# Decision tree
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(random_state=42, class_weight="balanced")
tree_clf.fit(X_train, y_train.values.ravel())
y_pred = tree_clf.predict(X_test)
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average=None))
# [0.77181783 0.44888978 0.06554622 0.01574803]
# [0.75862451 0.46306232 0.07464115 0.01769912]
# 0.6591871553483838
# [0.76516431 0.45586592 0.06979866 0.01666667]

from sklearn.ensemble import ExtraTreesClassifier
extree_clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
extree_clf.fit(X_train, y_train.values.ravel())
y_pred = extree_clf.predict(X_test)
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average=None))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf_pf = GaussianNB()
# clf_pf.fit(X_train, y_train.values.ravel())
# y_pred = clf_pf.predict(X_test)
# print(precision_score(y_test, y_pred, average=None))
# print(recall_score(y_test, y_pred, average=None))
# print(accuracy_score(y_test, y_pred))
# print(f1_score(y_test, y_pred, average=None))
# [0.90315644 0.37230769 0.01533082 0.00216129]
# [0.26073076 0.00624226 0.05454545 0.92920354]
# 0.18345367577613259
# [0.40464516 0.01227865 0.0239345  0.00431256]

# Stochastic Gradient Descent (SGD) classifier
from sklearn.linear_model import SGDClassifier
sgd_clf = make_pipeline(
    StandardScaler(),
    SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
)
# sgd_clf.fit(X_train, y_train.values.ravel())
# y_pred = sgd_clf.predict(X_test)
# print(precision_score(y_test, y_pred, average=None))
# print(recall_score(y_test, y_pred, average=None))
# print(accuracy_score(y_test, y_pred))
# print(f1_score(y_test, y_pred, average=None))
# [0.73560244 0.75728891 0.05970149 0.22222222]
# [0.98323874 0.20635576 0.00382775 0.01769912]
# 0.7365871492550955
# [0.8415819  0.32433309 0.00719424 0.03278689]

from sklearn.ensemble import BaggingClassifier
bagging = make_pipeline(StandardScaler(), BaggingClassifier(SGDClassifier(max_iter=1000, tol=1e-3, random_state=42), max_samples=0.5, max_features=0.5))
bagging.fit(X_train, y_train.values.ravel())
y_pred = bagging.predict(X_test)
# print(precision_score(y_test, y_pred, average=None))
# print(recall_score(y_test, y_pred, average=None))
# print(accuracy_score(y_test, y_pred))
# print(f1_score(y_test, y_pred, average=None))

from sklearn.ensemble import AdaBoostClassifier
Ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
Ada_clf.fit(X_train, y_train.values.ravel())
y_pred = Ada_clf.predict(X_test)

print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))
print(accuracy_score(y_train, Ada_clf.predict(X_train)))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average=None))
# [0.73120026 0.78025952 0.         0.        ]
# [0.9880055  0.18922823 0.         0.        ]
# 0.7347134631203729
# [0.84042282 0.30458792 0.         0.        ]

from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train, y_train)
X_train, y_train.values.ravel())
y_pred = gb_clf.predict(X_test)
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))
print(accuracy_score(y_train, Ada_clf.predict(X_train)))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average=None))




from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# SVC_clf = make_pipeline(StandardScaler(), LinearSVC(random_state=42, tol=1e-5))
# SVC_clf.fit(X_train, y_train.values.ravel())
# y_pred = SVC_clf.predict(X_test)
# print(precision_score(y_test, y_pred, average=None))
# print(recall_score(y_test, y_pred, average=None))
# print(accuracy_score(y_test, y_pred))
# print(f1_score(y_test, y_pred, average=None))
# [0.75308448 0.68640742 0.         0.        ]
# [0.95541415 0.29777136 0.         0.        ]
# 0.7443713249855285
# [0.84226883 0.41535638 0.         0.        ]
# [0.84226883 0.41535638 0.         0.        ]

from sklearn.svm import SVC
svc_clf = make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=42))
svc_clf.fit(X_train, y_train.values.ravel())
y_pred = svc_clf.predict(X_test)
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average=None))


from sklearn.linear_model import Perceptron
perc_clf = make_pipeline(
    StandardScaler(),
    Perceptron(tol=1e-3, random_state=42)
)
perc_clf.fit(X_train, y_train.values.ravel())
y_pred = perc_clf.predict(X_test)
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average=None))
# [0.76423859 0.54244881 0.14886731 0.        ]
# [0.86810482 0.39357202 0.04401914 0.        ]
# 0.7133717210492643
# [0.8128672  0.45617077 0.06794682 0.        ]

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train.values.ravel())
y_pred = lda_clf.predict(X_test)
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average=None))
# [0.75763108 0.6658318  0.13893654 0.02916667]
# [0.94760997 0.26057573 0.15502392 0.12389381]
# 0.7307071261006002
# [0.84203787 0.37456433 0.14654003 0.04721754]

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda_clf = QuadraticDiscriminantAnalysis()
qda_clf.fit(X_train, y_train.values.ravel())
y_pred = qda_clf.predict(X_test)
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average=None))
# [0.89426657 0.37586207 0.00622744 0.00204032]
# [0.02662735 0.00562319 0.06602871 0.95575221]
# 0.022651799043353744
# [0.05171486 0.01108061 0.01138144 0.00407194]

# Logistic
# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state = 2)
# X_train_res, y_train_res = sm.fit_sample(X_train, y_train.values.ravel())

# data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler
# fit scaler on training data
norm = MinMaxScaler().fit(X_train)
# transform training data
X_train_norm = norm.transform(X_train)
# transform testing dataabs
X_test_norm = norm.transform(X_test)

from sklearn.linear_model import LogisticRegression
# logisticRegr = LogisticRegression(class_weight="balanced", random_state=42)
# # logisticRegr.fit(X_train_res, y_train_res.values.ravel())
# logisticRegr.fit(X_train_norm, y_train.values.ravel())
# y_pred = logisticRegr.predict(X_test_norm)
# print(precision_score(y_test, y_pred, average=None))
# print(recall_score(y_test, y_pred, average=None))
# print(accuracy_score(y_test, y_pred))
# print(f1_score(y_test, y_pred, average=None))
# [0.73201946 0.52632639 0.         0.        ]
# [0.90419918 0.26970697 0.         0.        ]
# 0.7008957133717211
# [0.80905006 0.35665314 0.         0.        ]

# class_weight="balanced"
# [0.83779585 0.3945555  0.05053574 0.00431704]
# [0.56192355 0.35740817 0.30239234 0.44247788]
# 0.49719708740821983
# [0.67267371 0.37506429 0.08659907 0.00855066]

# class_weight="balanced" with normalization
# [0.88691673 0.41712285 0.05476567 0.00894712]
# [0.56722242 0.53461618 0.29856459 0.49557522]
# 0.5531944063613929
# [0.69192698 0.46861717 0.09255414 0.0175769 ]
from sklearn.linear_model import LogisticRegressionCV
logisticRegre = LogisticRegressionCV(class_weight="balanced", solver="sag", random_state=42)
logisticRegre.fit(X_train_norm, y_train.values.ravel())
y_pred = logisticRegre.predict(X_test_norm)
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average=None))

pdb.set_trace()


from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(
    estimators=[('forest', forest_clf), ('tree', tree_clf), ('clf', clf_pf), ('sdg', sgd_clf)],
    voting='hard'
)
voting_clf.fit(X_train, y_train.values.ravel())
y_pred = voting_clf.predict(X_test)
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average=None))

# # Randomized Search CV
# from sklearn.metrics import precision_score, make_scorer
# # Create a precision scorer
# precision = make_scorer(precision_score)
# # Finalize the random search
# rfc = RandomForestClassifier(random_state=42)
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# #min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# #min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                #'min_samples_split': min_samples_split,
#                #'min_samples_leaf': min_samples_leaf,
#                #'bootstrap': bootstrap}
# rs = RandomizedSearchCV(
#     estimator=rfc,
#     param_distributions=random_grid,
#     n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1
# )
# rs.fit(X_train, y_train.values.ravel())
# print the mean test scores:
# print('The accuracy for each run was: {}.'.format(rs.cv_results_['mean_test_score']))
# print the best model score:
# print('The best accuracy for a single model was: {}'.format(rs.best_score_))

import pdb
pdb.set_trace()


### Classification [1 2 2b 3]
## ask for precision, recall, accuracy, F1 Score
## Precision: precision = TP/ (TP + FP)
from sklearn.metrics import precision_score
precision_score1 = precision_score(y_test, y_pred, average=None)
print(precision_score1)
# [0.91436612 0.76649323 0.5126173  0.1013986  0.04545455]

## recall/ sensitivity/ true positive rate = TP/ (TP + FN)
from sklearn.metrics import recall_score
recall_score1 = recall_score(y_test, y_pred, average=None)
print(recall_score1)
# [0.96152762 0.8304816  0.42248001 0.02911647 0.00961538]

## accuracy = (TP + TN)/ Total
from sklearn.metrics import accuracy_score
accuracy_score1 = accuracy_score(y_test, y_pred)
print(accuracy_score1)
# 0.7237847795007148

## F1 Score = F = 2/ (1/precision + 1/recall)
from sklearn.metrics import f1_score
f1_score1 = f1_score(y_test, y_pred, average=None)
print(f1_score1)
# [0.93735403 0.79720546 0.46320433 0.04524181 0.01587302]

# View confusion matrix for test data and predictions
matrix = confusion_matrix(y_test, y_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

import pdb
# pdb.set_trace()

"""
# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10}, cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['1', '2', '2b', '3']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()
"""




# Important features:
col = X_train.columns
rank = pd.DataFrame(data={'colname':col, 'importance':forest_clf.feature_importances_})

# Variables should be removed:  INJURIES, SERIOUSINJURIES, FATALITIES
rank.sort_values('importance', ascending=False)

# Set threshold
mask_feature = forest_clf.feature_importances_ > 0.1

# Apply the mask to the feature dataset X
reduced_X = X.loc[:, mask_feature]
