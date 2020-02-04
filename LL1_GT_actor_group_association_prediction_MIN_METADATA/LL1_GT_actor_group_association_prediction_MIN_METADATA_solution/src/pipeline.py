# coding: utf-8

"""
# Proof of concept
# Solving the GT problem as a purely timeseries problem and not as a graph time-series problem (using ARIMA model)
# Here, we ignore the structural aspects of the graph
# We treat each edge weight as a signal and see how it evolves over time. 
# We observe from [1, 4000] time steps for each edge and predict for future [4001, 4210] steps

# refers to the following files from the dataset here: https://gitlab.datadrivendiscovery.org/d3m/datasets/tree/master/seed_datasets_current/LL1_GT_actor_group_association_prediction_MIN_METADATA/LL1_GT_actor_group_association_prediction_MIN_METADATA_dataset
# 1. tables/learningData.csv
# 2. graphs/edgeList.csv

see the Jupyter Notebook for more details
"""

import os, json
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.iolib.smpickle import load_pickle, save_pickle
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
import warnings
import numpy, random
from collections import OrderedDict
from d3mds import D3MDS
from arima import DO_ARIMA
warnings.filterwarnings('ignore')

TARGET_COL = 'association_strength'

def lddf_edges_merge_and_save(df, edges, type):
	mddfs = []
	for i, t in enumerate(df.timestep.unique()):
		lddf = df[df.timestep==t][['timestep', 'edgeID', 'association_strength']]
		lddf.drop_duplicates(subset='edgeID', inplace=True)
		eddf = edges[['edgeID']]
		mddf = lddf.merge(right=eddf, on='edgeID', how='outer')
		mddf = mddf.drop_duplicates(keep='last')
		# display(mddf.head(), mddf.tail())
		mddf['timestep']=[t]*len(mddf)
		mddf = mddf.fillna(0.0)
		mddfs.append(mddf)

	bigdf = pd.concat(mddfs, axis=0)
	bigdf.to_csv('timeseries_%s.csv'%type, index=None)
	print(bigdf.shape)


# set the path and ensure that paths exist
here = os.path.dirname(os.path.abspath(__file__))
dspath = os.path.join(here, '..', '..', 'LL1_GT_actor_group_association_prediction_MIN_METADATA_dataset')
prpath = os.path.join(here, '..', '..', 'LL1_GT_actor_group_association_prediction_MIN_METADATA_problem')
solpath = os.path.join(here, '..')
assert os.path.exists(dspath)
assert os.path.exists(prpath)

# get train and test data
d3mds = D3MDS(dspath, prpath)

###############################################
prep_data = False
if prep_data:

	train_data = d3mds.get_train_data()
	train_data[TARGET_COL] = d3mds.get_train_targets()

	test_data = d3mds.get_test_data()
	test_data[TARGET_COL] = d3mds.get_test_targets()

	print(train_data.shape, test_data.shape)

	# join the train_/test_data and edgeList data into a consolidated table
	edges = pd.read_csv(os.path.join(dspath, 'graphs', 'edgeList.csv'))
	print(edges.shape)

	lddf_edges_merge_and_save(train_data, edges, 'train')
	lddf_edges_merge_and_save(test_data, edges, 'test')
###############################################

# load train and test timeseries data
train_data = pd.read_csv('timeseries_train.csv')
test_data = pd.read_csv('timeseries_test.csv')
print(train_data.shape, test_data.shape)

###############################################
enable_fit = False
if enable_fit:
	arima = DO_ARIMA(
		train_data, 
		test_data, 
		endo_var=TARGET_COL,
		groupby_var='edgeID',
		index_col='timestep',
		use_exo_vars=False)

	arima.fit()
	save_pickle(self.fitted_models, os.path.join(here, 'fitted_models.pickle'))
else:
	fitted_models = load_pickle(os.path.join(here, 'fitted_models.pickle'))
###############################################


###############################################
test_predictions = []
for edgeID in test_data['edgeID'].unique():
	print(edgeID, end=',')
	x_test = test_data[test_data['edgeID']==edgeID]['association_strength']
	x_test.index = test_data[test_data['edgeID']==edgeID]['timestep']
	
	model_fit = fitted_models[edgeID]
	output = model_fit.forecast(steps=len(x_test))
	yhat = output[0]
	yhat = np.where(yhat<0, 0, yhat)
	yhat_df = pd.DataFrame(yhat, columns=['pred_value'])
	yhat_df['edgeID'] = [edgeID]*len(yhat_df)
	yhat_df.index = x_test.index
	test_predictions.append(yhat_df)
test_predictions_df = pd.concat(test_predictions)
###############################################

score_df = test_data.merge(right=test_predictions_df, on=['timestep', 'edgeID'], how='inner')
score_df.reset_index(inplace=True, drop=True)

test_data = d3mds.get_test_data().reset_index(inplace=False)
print(test_data.shape)
print(test_data.tail())

merged_df = score_df.merge(right=test_data, on=['timestep', 'edgeID'], how='inner')
print(merged_df.shape)
print(merged_df.tail())

rmse = sqrt(mean_squared_error(merged_df[TARGET_COL], merged_df['pred_value']))
print('RMSE:', rmse)

predictions_df = merged_df[['pred_value']]
predictions_df.columns = [TARGET_COL]
predictions_df.index = merged_df['d3mIndex']
predictions_df.to_csv(os.path.join(here, '..', 'predictions.csv'))

scores = pd.DataFrame(columns=['metric','value'], data=[["rootMeanSquaredError", rmse]])
scores.to_csv(os.path.join(here, '..', 'scores.csv'), index=None)