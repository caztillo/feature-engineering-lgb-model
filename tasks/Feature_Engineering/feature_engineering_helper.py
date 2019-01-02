import pandas as pd
import numpy as np


class Feature_Engineering_Helper:

	# Creating sales lag features
	def create_sales_lag_feats(df, gpby_cols, target_col, lags):
		gpby = df.groupby(gpby_cols)
		for i in lags:
			df['_'.join([target_col, 'lag', str(i)])] = \
			gpby[target_col].shift(i).values + np.random.normal(scale=1.6, size=(len(df),))
		return df

	# Creating sales rolling mean features
	def create_sales_rmean_feats(df, gpby_cols, target_col, windows, min_periods=2, 
							 shift=1, win_type=None):
		gpby = df.groupby(gpby_cols)
		for w in windows:
			df['_'.join([target_col, 'rmean', str(w)])] = \
			gpby[target_col].shift(shift).rolling(window=w, 
				min_periods=min_periods,
				win_type=win_type).mean().values +\
			np.random.normal(scale=1.6, size=(len(df),))
		return df


	# Creating sales exponentially weighted mean features
	def create_sales_ewm_feats(df, gpby_cols, target_col, alpha=[0.9], shift=[1]):
		gpby = df.groupby(gpby_cols)
		for a in alpha:
			for s in shift:
				df['_'.join([target_col, 'lag', str(s), 'ewm', str(a)])] = \
				gpby[target_col].shift(s).ewm(alpha=a).mean().values
		return df

	def one_hot_encoder(df, ohe_cols=['store','item','dayofmonth','dayofweek','month','weekofyear']):
		'''
		One-Hot Encoder function
		'''
		print('Creating OHE features..\nOld df shape:{}'.format(df.shape))
		df = pd.get_dummies(df, columns=ohe_cols)
		print('New df shape:{}'.format(df.shape))
		return df
