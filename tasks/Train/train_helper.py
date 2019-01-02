import pandas as pd
import numpy as np
import time
import lightgbm as lgb

class Train_Helper:

	@staticmethod
	def smape(preds, target):
	    '''
	    Function to calculate SMAPE
	    '''
	    n = len(preds)
	    masked_arr = ~((preds==0)&(target==0))
	    preds, target = preds[masked_arr], target[masked_arr]
	    num = np.abs(preds-target)
	    denom = np.abs(preds)+np.abs(target)
	    smape_val = (200*np.sum(num/denom))/n
	    return smape_val

	@staticmethod
	def lgbm_smape(preds, train_data):
	    '''
	    Custom Evaluation Function for LGBM
	    '''
	    labels = train_data.get_label()
	    smape_val = Train_Helper.smape(np.expm1(preds), np.expm1(labels))
	    return 'SMAPE', smape_val, False

	@staticmethod
	def lgb_validation(params, lgbtrain, lgbval, X_val, Y_val, verbose_eval):
	    t0 = time.time()
	    evals_result = {}
	    model = lgb.train(params, lgbtrain, num_boost_round=params['num_boost_round'], 
	                      valid_sets=[lgbtrain, lgbval], feval=Train_Helper.lgbm_smape, 
	                      early_stopping_rounds=params['early_stopping_rounds'], 
	                      evals_result=evals_result, verbose_eval=verbose_eval)
	    print(model.best_iteration)
	    print('Total time taken to build the model: ', (time.time()-t0)/60, 'minutes!!')
	    pred_Y_val = model.predict(X_val, num_iteration=model.best_iteration)
	    pred_Y_val = np.expm1(pred_Y_val)
	    Y_val = np.expm1(Y_val)
	    val_df = pd.DataFrame(columns=['true_Y_val','pred_Y_val'])
	    val_df['pred_Y_val'] = pred_Y_val
	    val_df['true_Y_val'] = Y_val
	    print(val_df.shape)
	    print(val_df.sample(5))
	    print('SMAPE for validation data is:{}'.format(Train_Helper.smape(pred_Y_val, Y_val)))
	    return model, val_df

	@staticmethod
	def lgb_train(params, lgbtrain_all, X_test, num_round):
		t0 = time.time()
		model = lgb.train(params, lgbtrain_all, num_boost_round=num_round, feval=Train_Helper.lgbm_smape)
		test_preds = model.predict(X_test, num_iteration=num_round)
		print('Total time taken in model training: ', (time.time()-t0)/60, 'minutes!')
		return model, test_preds