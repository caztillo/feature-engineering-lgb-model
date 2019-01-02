import luigi
import pandas as pd
import numpy as np
from tasks.Preprocessing.preprocessing_task import ProcessTrainAndTestData

class FeatureEngineering(luigi.Task):
	def requires(self):

		return ProcessTrainAndTestData()

	def run(self):
		df = pd.read_csv(self.input().path, parse_dates=['date'])

		# Extracting date features
		df['dayofmonth'] = df.date.dt.day
		df['dayofyear'] = df.date.dt.dayofyear
		df['dayofweek'] = df.date.dt.dayofweek
		df['month'] = df.date.dt.month
		df['year'] = df.date.dt.year
		df['weekofyear'] = df.date.dt.weekofyear
		df['is_month_start'] = (df.date.dt.is_month_start).astype(int)
		df['is_month_end'] = (df.date.dt.is_month_end).astype(int)

		# Sorting the dataframe by store then item then date
		df.sort_values(by=['store','item','date'], axis=0, inplace=True)

		df['sales'] = np.log1p(df.sales.values)

		# For validation 
		# We can choose last 3 months of training period(Oct, Nov, Dec 2017) as our validation set to gauge the performance of the model.
		# OR to keep months also identical to test set we can choose period (Jan, Feb, Mar 2017) as the validation set.
		# Here we will go with the latter choice.
		masked_series = (df.year==2017) & (df.month.isin([1,2,3]))
		masked_series2 = (df.year==2017) & (~(df.month.isin([1,2,3])))
		df.loc[(masked_series), 'train_or_test'] = 'val'
		df.loc[(masked_series2), 'train_or_test'] = 'no_train'
		print('Train shape: {}'.format(df.loc[df.train_or_test=='train',:].shape))
		print('Validation shape: {}'.format(df.loc[df.train_or_test=='val',:].shape))
		print('No train shape: {}'.format(df.loc[df.train_or_test=='no_train',:].shape))
		print('Test shape: {}'.format(df.loc[df.train_or_test=='test',:].shape))

		df.to_csv(self.output().path,index=False)

	def output(self):
		return luigi.LocalTarget("tmp/feature_engineering.csv")

