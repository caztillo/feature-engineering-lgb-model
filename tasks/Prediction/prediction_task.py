import luigi
import pandas as pd
import numpy as np
from tasks.Train.train_task import Train
from tasks.Feature_Engineering.feature_engineering_task import FeatureEngineering
from tasks.Feature_Engineering.feature_engineering_helper import Feature_Engineering_Helper
from tasks.Train.train_helper import Train_Helper
import lightgbm as lgb
from sklearn.externals import joblib

class Prediction(luigi.Task):
	def requires(self):
		return Train()

	def run(self):
		df = pd.read_csv("tmp/feature_engineering.csv", parse_dates=['date'])

		# Creating sales lag, rolling mean, rolling median, ohe features of the above train set
		df_whole = Feature_Engineering_Helper.create_sales_lag_feats(df, gpby_cols=['store','item'], target_col='sales', 
		                                  lags=[91,98,105,112,119,126,182,364,546,728])
		df_whole = Feature_Engineering_Helper.create_sales_rmean_feats(df_whole, gpby_cols=['store','item'], 
		                                    target_col='sales', windows=[364,546], 
		                                    min_periods=10, win_type='triang')
		# df = create_sales_rmed_feats(df, gpby_cols=['store','item'], target_col='sales', 
		#                              windows=[364,546], min_periods=2) #98,119,
		df_whole = Feature_Engineering_Helper.create_sales_ewm_feats(df_whole, gpby_cols=['store','item'], target_col='sales', 
		                                  alpha=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5], 
		                                  shift=[91,98,105,112,119,126,182,364,546,728])

		# # Creating sales monthwise aggregated values
		# agg_df = create_sales_agg_monthwise_features(df.loc[~(df.train_or_test=='test'), :], 
		#                                              gpby_cols=['store','item','month'], 
		#                                              target_col='sales', 
		#                                              agg_funcs={'mean':np.mean, 
		#                                              'median':np.median, 'max':np.max, 
		#                                              'min':np.min, 'std':np.std})

		# # Joining agg_df with df
		# df = df.merge(agg_df, on=['store','item','month'], how='left')

		# One-Hot Encoding
		df_whole = Feature_Engineering_Helper.one_hot_encoder(df_whole, ohe_cols=['store','item','dayofweek','month']) 
		#'dayofmonth',,'weekofyear'

		# Final train and test datasets
		test = df_whole.loc[df_whole.train_or_test=='test', :]
		train = df_whole.loc[~(df_whole.train_or_test=='test'), :]
		print('Train shape:{}, Test shape:{}'.format(train.shape, test.shape))

		avoid_cols = ['date', 'sales', 'train_or_test', 'id', 'year']
		cols = [col for col in train.columns if col not in avoid_cols]

		# LightGBM dataset
		lgbtrain_all = lgb.Dataset(data=train.loc[:,cols].values, 
                           label=train.loc[:,'sales'].values.reshape((-1,)), 
                           feature_name=cols)

		# LightGBM parameters
		lgb_params = {'task':'train', 'boosting_type':'gbdt', 'objective':'regression', 
              'metric': {'mae'}, 'num_leaves': 10, 'learning_rate': 0.02, 
              'feature_fraction': 0.8, 'max_depth': 5, 'verbose': 0, 
              'num_boost_round':15000, 'nthread':-1}

		model = joblib.load("tmp/train_model.joblib") 

        # Training lgb model on whole data(train+val)
		lgb_model, test_preds = Train_Helper.lgb_train(lgb_params, lgbtrain_all, test.loc[:,cols].values, model.best_iteration)
		print('test_preds shape:{}'.format(test_preds.shape))

		# Create submission
		sub = test.loc[:,['id','sales']]
		sub['sales'] = np.expm1(test_preds)
		sub['id'] = sub.id.astype(int)
		sub.head()

		

		sub.to_csv(self.output().path,index=False)

	def output(self):
		return luigi.LocalTarget("tmp/submission.csv")

