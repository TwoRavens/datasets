# coding: utf-8

"""
Created on Fri Feb 28 2020
@author: Swaroop Vattam

Works best if run with 4 or more GPUs
"""

import sys, os, pickle
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import pandas as pd
from d3mds import D3MDataset, D3MProblem, D3MDS
from mlutils.preprocessing import Drift_thresholder
from mlutils.encoding import NA_encoder, Categorical_encoder
from mlutils.model.classification import Clf_feature_selector, Classifier
from mlutils.prediction import Predictor
from mlutils.preprocessing import Reader as Munger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

here = os.path.dirname(os.path.abspath(__file__))

dspath = os.path.join(here, '..', '..', 'LL1_MITLL_synthetic_vora_E_dataset')
prpath = os.path.join(here, '..', '..', 'LL1_MITLL_synthetic_vora_E_problem')
solpath = os.path.join(here, '..')
assert os.path.exists(dspath)
assert os.path.exists(prpath)

d3mds = D3MDS(dspath, prpath)

target_name = d3mds.problem.get_targets()[0]['colName']

X_train = d3mds.get_train_data() # read train data
X_train[target_name] = d3mds.get_train_targets().ravel() # read train targets and merge it with the train data dataFrame
X_train.to_csv('X_train.csv') # save the train dataFrame

X_test = d3mds.get_test_data() # read test data
X_test.to_csv('X_test.csv') # save the test dataFrame

# clean the train and test data
mgr = Munger(sep = ',')
df = mgr.train_test_split(['X_train.csv', 'X_test.csv'], target_name)

# calculate the drift and threshold data (used both train and test data)
dft = Drift_thresholder()
data = dft.fit_transform(df)

# process numerical columns in train data
na_encoder = NA_encoder(numerical_strategy=0)
data['train'] = na_encoder.fit_transform(data['train'])

# process/encode categorical columns in trian data
ce_encoder = Categorical_encoder(strategy='random_projection')
data['train'] = ce_encoder.fit_transform(data['train'], data['target'])

# do feature selection in train data
fs_selector = Clf_feature_selector(strategy='rf_feature_importance', threshold=0.3)
data['train'] = fs_selector.fit_transform(data['train'], data['target'])

# fit an estimator
estimator = Classifier(strategy= "LightGBM")
estimator.fit(data['train'], data['target'])

# perform above processing steps on test data
data['test'] = na_encoder.transform(data['test'])
data['test'] = ce_encoder.transform(data['test'])
data['test'] = fs_selector.transform(data['test'])

y_pred = estimator.predict(data['test']) # make predictions

# decode the predictions
fhand = open(os.path.join('save', "target_encoder.obj"), 'rb')
enc = pickle.load(fhand)
fhand.close()
y_pred = enc.inverse_transform(y_pred)
y_pred_df = pd.DataFrame(index=X_test.index, data=y_pred, columns=[target_name])
y_pred_df.to_csv(os.path.join('.','predictions.csv'))


# read the truth target values in test data
y_test = d3mds.get_test_targets().ravel()

print(classification_report(y_test, y_pred)) # classification report

# compute the f1 score
f_score = f1_score(y_test, y_pred, pos_label=1)
scoresdf = pd.DataFrame(columns=['metric','value','randomState'])
scoresdf.loc[len(scoresdf)]=['f1', f_score, 'N/A']
scoresdf.to_csv(os.path.join('.','scores.csv'))