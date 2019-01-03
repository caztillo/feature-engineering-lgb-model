import luigi
import pandas as pd
import numpy as np
from tasks.Feature_Engineering.feature_engineering_task import FeatureEngineering
from tasks.Feature_Engineering.feature_engineering_helper import Feature_Engineering_Helper
from tasks.Train.train_helper import Train_Helper
import lightgbm as lgb
from sklearn.externals import joblib

class Train(luigi.Task):
	def requires(self):

		return FeatureEngineering()

	def run(self):
		df = pd.read_csv(self.input().path)

		# Converting sales of validation period to nan so as to resemble test period
		train = df.loc[df.train_or_test.isin(['train','val']), :]
		Y_val = train.loc[train.train_or_test=='val', 'sales'].values.reshape((-1))
		Y_train = train.loc[train.train_or_test=='train', 'sales'].values.reshape((-1))
		train.loc[train.train_or_test=='val', 'sales'] = np.nan

		# # Creating sales lag, rolling mean, rolling median, ohe features of the above train set
		train = Feature_Engineering_Helper.create_sales_lag_feats(train, gpby_cols=['store','item'], target_col='sales', 
									   lags=[91,98,105,112,119,126,182,364,546,728])

		train = Feature_Engineering_Helper.create_sales_rmean_feats(train, gpby_cols=['store','item'], 
										 target_col='sales', windows=[364,546], 
										 min_periods=10, win_type='triang') #98,119,91,182,

		# # train = create_sales_rmed_feats(train, gpby_cols=['store','item'], 
		# #                                 target_col='sales', windows=[364,546], 
		# #                                 min_periods=10, win_type=None) #98,119,91,182,

		train = Feature_Engineering_Helper.create_sales_ewm_feats(train, gpby_cols=['store','item'], 
			target_col='sales', 
			alpha=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5], 
			shift=[91,98,105,112,119,126,182,364,546,728])

		# # Joining agg_df with train
		# train = train.merge(agg_df, on=['store','item','month'], how='left')

		# One-Hot Encoding 
		train = Feature_Engineering_Helper.one_hot_encoder(train, ohe_cols=['store','item','dayofweek','month']) 
		#,'dayofmonth','weekofyear'

		# Final train and val datasets
		val = train.loc[train.train_or_test=='val', :]
		train = train.loc[train.train_or_test=='train', :]
		print('Train shape:{}, Val shape:{}'.format(train.shape, val.shape))
		avoid_cols = ['date', 'sales', 'train_or_test', 'id', 'year']
		cols = [col for col in train.columns if col not in avoid_cols]
		print('No of training features: {} \nAnd they are:{}'.format(len(cols), cols))

		# LightGBM parameters
		lgb_params = {'task':'train', 'boosting_type':'gbdt', 'objective':'regression', 
			  'metric': {'mae'}, 'num_leaves': 10, 'learning_rate': 0.02, 
			  'feature_fraction': 0.8, 'max_depth': 5, 'verbose': 0, 
			  'num_boost_round':15000, 'early_stopping_rounds':200, 'nthread':-1}

		# Creating lgbtrain & lgbval
		lgbtrain = lgb.Dataset(data=train.loc[:,cols].values, label=Y_train, 
							   feature_name=cols)
		lgbval = lgb.Dataset(data=val.loc[:,cols].values, label=Y_val, 
							 reference=lgbtrain, feature_name=cols)

		# Training lightgbm model and validating
		model, val_df = Train_Helper.lgb_validation(lgb_params, lgbtrain, lgbval, val.loc[:,cols].values, 
							   Y_val, verbose_eval=500)

		# Let's see top 25 features as identified by the lightgbm model.
		print("Features importance...")
		gain = model.feature_importance('gain')
		feat_imp = pd.DataFrame({'feature':model.feature_name(), 
								 'split':model.feature_importance('split'), 
								 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
		print('Top 25 features:\n', feat_imp.head(25))

		joblib.dump(model, "tmp/train_model.joblib")

		val_df.to_csv(self.output().path,index=False)

	def output(self):
		return luigi.LocalTarget("tmp/train.csv")

