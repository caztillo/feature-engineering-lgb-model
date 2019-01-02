import luigi
import pandas as pd

class CollectTrainData(luigi.Task):
	def output(self):
		return luigi.LocalTarget("input/train.csv")

class CollectTestData(luigi.Task):
	def output(self):
		return luigi.LocalTarget("input/test.csv")

class ProcessTrainAndTestData(luigi.Task):
	def requires(self):

		return {
		'train': CollectTrainData(),
		'test': CollectTestData()
		}

	def run(self):
		train = pd.read_csv(self.input()['train'].path, parse_dates=['date'])
		test = pd.read_csv(self.input()['test'].path, parse_dates=['date'])

		train['train_or_test'] = 'train'
		test['train_or_test'] = 'test'

		print('Train shape:{}, Test shape:{}'.format(train.shape, test.shape))

		df = pd.concat([train,test], sort=False)
		print('Combined df shape:{}'.format(df.shape))
		df.to_csv(self.output().path,index=False)

	def output(self):
		return luigi.LocalTarget("tmp/combined.csv")
