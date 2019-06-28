import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.linear_model import ARDRegression
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet

df = pd.read_csv('new.csv')
x = df[['int_rate','emp_length','annual_inc','dti','installment','home_ownership_OWN','home_ownership_RENT','home_ownership_MORTGAGE','purpose_wedding','purpose_vacation','purpose_renewable_energy','purpose_moving','home_ownership_OTHER','purpose_medical','purpose_major_purchase','purpose_house','purpose_home_improvement','purpose_educational','purpose_car','purpose_small_business','verification_status_Not Verified','term_ 36 months','term_ 60 months']]
#'purpose_wedding','purpose_vacation','purpose_renewable_energy','purpose_moving'
y = df[['funded_amnt_inv']]/10000
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)
print(x_train.shape)
print(x_test.shape)
# linearReg = linear_model.ElasticNet(alpha=0.001,l1_ratio=0.1)
# linearReg = linear_model.LinearRegression()
# linearReg = linear_model.LassoCV(alphas=[0.001,0.1, 0.01, 0.5, 1, 3, 5, 7, 10, 20, 100], cv=5)
linearReg = linear_model.Lasso(alpha=0.01)
# linearReg = linear_model.Ridge(alpha=0.001)
# linearReg.fit(x_train,y_train)
# linearReg = Ridge()
linearReg.fit(x_train,y_train.values.ravel())
# print(linearReg.alpha_)
print(linearReg.intercept_)
print(linearReg.coef_)
y_pred = linearReg.predict(x_test)
Y_pred = cross_val_predict(linearReg, x, y,cv=100)
print('MSE:')
print(metrics.mean_squared_error(y_test,y_pred))
print('RMSE:')
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

# Y_pred = cross_val_predict(linearReg, x, y, cv=50)
print("10折交叉验证MSE:", metrics.mean_squared_error(y, Y_pred))
print("10折交叉验证RMSE:", np.sqrt(metrics.mean_squared_error(y, Y_pred)))

plt.figure()
plt.title("Model Star")
plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.ylim(0,3.5)
plt.grid(True)
plt.plot(y_test,y_pred,'r.')
plt.show()

