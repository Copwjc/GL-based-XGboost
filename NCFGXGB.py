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
import warnings
warnings.filterwarnings("ignore")


orders = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# params = [
# {'learning_rate': 0.181,'max_depth': 3,'alpha': 6.248014163607554e-07,'lambda': 3.666161662695617e-06}, 
# {'learning_rate': 0.086,'max_depth': 3,'alpha': 2.886812764315837e-07,'lambda': 1.2294082899296007e-07}, 
# {'learning_rate': 0.066,'max_depth': 4,'alpha': 0.2898262687925129,'lambda': 0.0033367001415999143}, 
# {'learning_rate': 0.076,'max_depth': 3,'alpha': 3.9945333441962694,'lambda': 3.6415021378537495e-07},
# {'learning_rate': 0.071,'max_depth': 3,'alpha': 0.031506923349484094,'lambda': 1.1562866585387308e-07},
# {'learning_rate': 0.066,'max_depth': 3,'alpha': 2.7433507561223237e-08,'lambda': 0.004646582626195847},
# {'learning_rate': 0.041,'max_depth': 4,'alpha': 0.0036167404545273526,'lambda': 1.1505800980709108e-05},
# {'learning_rate': 0.101,'max_depth': 3,'alpha': 9.374433708232626e-08,'lambda': 3.9340007207957334e-08},
# {'learning_rate': 0.066,'max_depth': 6,'alpha': 0.008329059650517832,'lambda': 0.001198146925936356},
# ]  


# params = [
# {'learning_rate':0.13,'max_depth': 6,'alpha': 3e-06,'lambda': 1.3579333841650756}, 
# {'learning_rate':0.053,'max_depth': 6,'alpha': 8e-08,'lambda': 2.4609306182904254}, 
# {'learning_rate':0.128,'max_depth': 5,'alpha': 1.5556198654766862,'lambda':0.0004596768817880325}, 
# {'learning_rate':0.074,'max_depth': 4,'alpha': 9e-5,'lambda':0.12496555079234584}, 
# {'learning_rate':0.059,'max_depth': 6,'alpha': 1.113262572673615,'lambda': 5.433497546812114}, 
# {'learning_rate':0.048,'max_depth': 6,'alpha': 9.5e-6,'lambda': 1.1393584205166276},
# {'learning_rate':0.0514,'max_depth': 6,'alpha': 5e-8,'lambda': 1.51288000208726}, 
# {'learning_rate':0.074,'max_depth': 6,'alpha': 9.5e-8,'lambda': 1.5142621703105723}, 
# {'learning_rate':0.097,'max_depth': 5,'alpha': 1.0464246514785462,'lambda':0.12168301246536571}
# ]

params = [

{'learning_rate': 0.104,'max_depth': 7,'alpha': 0.006681658919465169,'lambda': 9.985299665083456},
{'learning_rate': 0.053,'max_depth': 6,'alpha': 8e-08,'lambda': 2.4609306182904254},
{'learning_rate': 0.043, 'max_depth': 7, 'alpha': 0.18259309529937776, 'lambda': 6.293252225746293},
{'learning_rate': 0.049, 'max_depth': 6, 'alpha': 0.0016496409099327415, 'lambda': 1.0058555736582145e-06},
{'learning_rate': 0.059,'max_depth': 6,'alpha': 1.113262572673615,'lambda': 5.433497546812114},
{'learning_rate': 0.054, 'max_depth': 7, 'alpha': 4.063937296460924, 'lambda': 9.891243764543823},
{'learning_rate': 0.059, 'max_depth': 6, 'alpha': 2.4396127114082963e-08, 'lambda': 1.342530065453554e-07},
{'learning_rate': 0.049, 'max_depth': 8, 'alpha': 3.0664573979627927, 'lambda': 7.347817706508894}, 
{'learning_rate': 0.068,'max_depth': 7,'alpha': 4.73850166229978e-08,'lambda': 5.841539293329681},

]

X = pd.read_csv('dataset/x.csv', index_col = [0])
Y = pd.read_csv('dataset/y.csv', index_col = [0])

x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.25, random_state=2025)
test_num = x_valid.shape[0]
train_num = x_train.shape[0]

k = 10

def coefficient(k,ordered):
    t = 1
    for i in range(0,k):
        t *= (ordered-i)
    t *= (-1)**k
    t = t/gamma(k+1)
    return t
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

rs = []
for i,ordered in enumerate(orders):

    pa = params[i]
    
    coef = []
    pa = params[i]
    for i in range(0,k):
        coef.append(coefficient(i,ordered))
    
    coef = [coef] * train_num
    
    xgbr = xgb.XGBRegressor(**pa) 
    evals_results = {}

    xgbr.set_params(**{'objective': custom_train}, 
        n_estimators = 400)

    # fitting model 
    xgbr.fit(x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric = ['mae','rmse'],
        verbose = 0)

    pred_y = xgbr.predict(x_valid)
    pd.DataFrame(pred_y).to_csv(f'prd/{ordered}xgbpred.csv')
    # y_valid = y_valid.to_numpy().reshape(test_num,1)


    a = xgbr.evals_result_.pop('validation_0')
    out = pd.DataFrame(a['rmse'])
    out1 = pd.DataFrame(a['mae'])
    out.to_csv(f'output/{ordered}NCFGXGBM(R).csv')
    out1.to_csv(f'output/{ordered}NCFGXGBM(M).csv')



    from sklearn.metrics import mean_absolute_percentage_error as mape
    rs.append([
        ordered,
        mae(y_valid, pred_y),
        r2_score(y_valid, pred_y),
        mape(y_valid, pred_y),
        mse(y_valid, pred_y)**(1/2)
    ])

print(rs)
# = pd.DataFrame(evals_result)
#a.to_excel('./1.xlsx')



#out = pd.DataFrame(a['rmse'])
#out1 = pd.DataFrame(pred_y)
#out.to_csv('output/fom.csv')
#out1.to_csv('output/fopred.csv')
#plt.plot(pred_y,label = 'pred')
#plt.plot(y_valid, label = 'true')
#plt.legend()
#plt.show()


# = pd.DataFrame(evals_result)
#a.to_excel('./1.xlsx')

    