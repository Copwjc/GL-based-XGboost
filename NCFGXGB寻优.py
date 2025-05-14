import pandas as pd
import optuna
import random
import xgboost as xgb
from xgboost import XGBRegressor
from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

orders = [0.6, 0.7, 0.8, 0.9]
ls = []
X = pd.read_csv('dataset/x.csv', index_col = [0])
Y = pd.read_csv('dataset/y.csv', index_col = [0])
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.25, random_state=2025)
test_num = x_valid.shape[0]
train_num = x_train.shape[0]
def f(x):
	return (x**2)/2
def ncasual(coef,x,f):
		global k
		p4 = []
		p5 = []
		for i in range(0,k):
			temp1 = f(x+i)
			temp2 = f(x-i)
			p4.append(temp1)
			p5.append(temp2)
		out = (sum(np.array(coef).T*(np.array(p5)-np.array(p4))))
		return out

def custom_train(y_true, y_pred):
	global t, epsilon
	residual = (y_true - y_pred).astype("float")
	grad = -ncasual(coef, residual, f)
	hess = np.where(residual<0, 2.0, 2.0)
	return grad, hess


for ordered in orders:
	k = 10
	def coefficient(k,ordered):
		t = 1
		for i in range(0,k):
			t *= (ordered-i)
		t *= (-1)**k
		t = t/gamma(k+1)
		return t

	#计算系数
	coef = []
	for i in range(0,k):
		coef.append(coefficient(i,ordered))
	coef = [coef] * train_num


	# default lightgbm model with sklearn api
	xgbr = xgb.XGBRegressor() 
	def objective(trial):
		pruning_callback = optuna.integration.LightGBMPruningCallback(trial, mse)
		params = {
			'learning_rate':trial.suggest_float('learning_rate', 0.001, 0.351, step = 0.001),
			'max_depth':trial.suggest_int('max_depth', 3, 20, step = 1), 
			"alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
			"lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
		}
		model = XGBRegressor(**params, n_estimators = 400, objective = custom_train)
		model.fit(x_train, y_train)
		pred_y = model.predict(x_valid)
		score = mae(y_valid, pred_y)
		return score

	study = optuna.create_study(direction="minimize", pruner = optuna.pruners.MedianPruner)

	study.optimize(objective, n_trials = 200)

	# print("Number of finished trials: ({})".format(len(study.trials)))

	# print("Best trial:")
	trial = study.best_trial
	print(trial.value)
	# print("  Value: {}".format(trial.value))

	# print("  Params: ")
	# for key, value in trial.params.items():
	# 	print("    {}: {}".format(key, value))
	ls.append([ordered, trial.params.items()])
print(ls)
		