#lib
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


#dataset
from google.colab import files
upload = files.upload()


dataset = pd.read_csv("FuelConsumption.csv")
dataset


x_variable = ['FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG','FUELCONSUMPTION_HWY' ]
y_variable = ['CO2EMISSIONS']


x = dataset[x_variable].values.reshape(-1,3)
y = dataset[y_variable].values.reshape(-1,1)

x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size = 0.2) # 80% for train, 20% for testing

print(x_train.shape)
print(x_test.shape)

multiple_reg = LinearRegression()

#training
model = multiple_reg.fit(x_train, y_train)


#finding intercept & slope(Coefficient)
Q_0 = model.intercept_
print(Q_0)
Q_1 = model.coef_
print(Q_1)



y_predict = model.predict(x_test)
print(y_predict)

x_assum1 = np.array([30,3,50]) 
x_assum1 = x_assum1.reshape(-1,1)

y_predict = y_predict = model.predict(x_test)
print(y_predict)


#graph == I know it wrong ??
plt.scatter(x_test[:, 0] , y_test, color = "Blue")
plt.plot(x_test, y_predict, color = "Red")
plt.show()



#===== MSE Start ===== 
MSE = mean_squared_error(y_test, y_predict)
print(f"This is MSE value: {MSE}")
#===== MSE end =====