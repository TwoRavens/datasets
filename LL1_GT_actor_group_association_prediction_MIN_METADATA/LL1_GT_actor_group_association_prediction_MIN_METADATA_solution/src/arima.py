import os, json,sys
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from math import sqrt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.iolib.smpickle import load_pickle
from pandas.plotting import autocorrelation_plot
import warnings
import numpy, random
from collections import OrderedDict
from statsmodels.graphics.tsaplots import plot_acf
from d3mds import D3MDataset, D3MProblem, D3MDS
warnings.filterwarnings('ignore')

# Custom ARIMA class
class DO_ARIMA:
    def __init__(self, 
                 train_data, 
                 test_data, 
                 endo_var, 
                 groupby_var,
                 index_col,
                 exo_vars = None, 
                 use_exo_vars = False):
        
        self.train_data = train_data
        self.test_data = test_data
        self.endo_var = endo_var
        self.exo_vars = exo_vars
        self.groupby_var = groupby_var
        self.index_col = index_col # should be a time index
        self.use_exo_vars = use_exo_vars
        
        # perform some basic checks
        assert set(train_data.columns) == set(test_data.columns)
        assert endo_var in train_data.columns
        if groupby_var: assert groupby_var in train_data.columns
        if exo_vars:
            for c in exo_vars: 
                assert c in train_data.columns
        
        self.fitted_models = {}
        self.test_predictions_df = None
        
    def add_noise(self, X):
        # adds gaussian noise to the input
        X_is_dataframe = True if type(X) == type(pd.DataFrame()) else False
        if X_is_dataframe:
            current_columns=X.columns
            current_index=X.index
        mu, sigma = 0, 0.01
        rng = np.random.RandomState(42)
        noise = rng.normal(mu, sigma, X.shape)
        X += noise
        if X_is_dataframe:
            X = pd.DataFrame(X, columns=current_columns, index=current_index)
        return X
        
                
    def fit(self, num_points_in_validation=1000):
        for i, groupid in enumerate(self.train_data[self.groupby_var].unique()):
            x = self.train_data[self.train_data[self.groupby_var]==groupid][self.endo_var]
            x.index = self.train_data[self.train_data[self.groupby_var]==groupid][self.index_col]
            if self.use_exo_vars:
                exos = self.train_data[self.train_data[self.groupby_var]==groupid][self.exo_vars]
                exos.index = self.train_data[self.train_data[self.groupby_var]==groupid][self.index_col]

            train_validation_split_point = len(x)-num_points_in_validation
            x_train = x[:train_validation_split_point]
            x_test = x[train_validation_split_point:]
            if self.use_exo_vars:
                exos_train = exos[:train_validation_split_point]
                exos_test = exos[train_validation_split_point:]
            
            history_endo = self.add_noise(np.array([x for x in x_train]).astype(float))
            if self.use_exo_vars:
                history_exos = self.add_noise(exos_train)
            
            predictions = list()
            for j, t in enumerate(x_test.index.values):
                obs_endo = x_test.iloc[j]
                if self.use_exo_vars:
                    obs_exo = exos_test.iloc[j]
                
                if not self.use_exo_vars:
                    model = ARIMA(history_endo, order=(1,1,0))
                else:
                    model = ARIMA(history_endo, exog=history_exos, order=(1,1,0)) # ARIMA model with exo
                
                model_fit = model.fit(disp=0)
                
                if not self.use_exo_vars:
                    output = model_fit.forecast(steps=1)
                else:
                    output = model_fit.forecast(steps=1, exog=obs_exo) 
                
                yhat = output[0][0]
                yhat = np.where(yhat<0, 0, yhat)
                predictions.append(yhat)
                
                # add observation in this cycle to the history of the next cycle
                history_endo = np.append(history_endo, obs_endo)
                if self.use_exo_vars:
                    history_exos = history_exos.append(obs_exo)
                
                print(groupid, j, 'predicted=%f, expected=%f' % (yhat, obs_endo))
                self.fitted_models[groupid] = model_fit
        
                
    def predict(self):
        test_predictions = []
        for i, groupid in enumerate(self.test_data[self.groupby_var].unique()):
            x_test = self.test_data[self.test_data[self.groupby_var]==groupid][self.endo_var]
            x_test.index = self.test_data[self.test_data[self.groupby_var]==groupid][self.index_col]
            
            if self.use_exo_vars:
                exos = self.test_data[self.test_data[self.groupby_var]==groupid][self.exo_vars]
                exos.index = self.test_data[self.test_data[self.groupby_var]==groupid][self.index_col]
                exos = self.add_noise(exos)
            
            model_fit = self.fitted_models[groupid]
            
            if not self.use_exo_vars:
                output = model_fit.forecast(steps=len(x_test))
            else:
                output = model_fit.forecast(steps=len(x_test), exog=exos)
                
            yhat = output[0]
            yhat = np.where(yhat<0, 0, yhat).round()
            yhat_df = pd.DataFrame(yhat, columns=['pred_value'])
            yhat_df[self.groupby_var] = [groupid]*len(yhat_df)
            yhat_df.index = x_test.index
            test_predictions.append(yhat_df)
            
        self.test_predictions_df = pd.concat(test_predictions)
            
                
    def score(self):
        score_df = self.test_data.merge(right=self.test_predictions_df, on=[self.index_col, self.groupby_var], how='inner')
        score_df.reset_index(inplace=True, drop=True)
        # display(score_df)
        rmse = sqrt(mean_squared_error(score_df[self.endo_var], score_df['pred_value']))
        mae = (mean_absolute_error(score_df[self.endo_var], score_df['pred_value']))
        
        # plot_acf(score_df[self.endo_var].values.squeeze(), lags=10); plt.show()
        # plot_acf(score_df['pred_value'].values.squeeze(), lags=10); plt.show()
        # autocorrelation_plot(score_df[self.endo_var]); plt.show()
        # autocorrelation_plot(score_df['pred_value']); plt.show()       

        return mae, rmse