#lib
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from google.colab import files 
upload = files.upload()


# == If you use it from google drive ==
#from google.colab import drive
#upload = drive.mount() 
# ============================


dataset = pd.read_csv("FuelConsumption.csv")
dataset

#Linear -make sure your dataset is clean and tidy (*****)


#Selecte the variable (x, y)

x_varaible = ['FUELCONSUMPTION_HWY'] #Variable X name
y_variable = ['CO2EMISSIONS'] #Variable y name

#== Using both for how many x value and y value == 


x = np.array(dataset[x_varaible]).reshape(-1,1)#check how many x variable we have 1st values is -1 and 1 is len of variable x 
y = np.array(dataset[y_variable]).reshape(-1,1)

# y_value = [i[0] for i in y]  #Make it to only one list

# x = dataset[x_varaible].values.reshape(-1,1)
# y = dataset[y_variable].values.reshape(-1,1)

#== Using both for how many x value and y value == 

#split the dataset
x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size = 0.2)  #80% for train, 20% for testing
y_test_value = [i[0]  for i in y_test]

#Build a model (simple linear)
lin_reg = LinearRegression()


#training
model = lin_reg.fit(x_train, y_train)


#finding intercept & slope(Coefficient)
Q_0 = model.intercept_
Q_1 = model.coef_
#just for understanding


y_hat = Q_0 + Q_1*x 



#== predict the whole dataset ==
y_predict = model.predict(x_test)
y_predict_value = [i[0]  for i in y_predict]

# y_predict_value =  [i[0] for i in y_predict]  #Make it to only one list

# print(y_predict)

# ========================


#== predict the single value ==
# x_assum = np.array([5]) # if 'FUELCONSUMPTION_HWY value is 250, what is the answer(CO2emission)?
# x_assum = x_assum.reshape(-1,1)

# y_predict = model.predict(x_assum)
# print(y_predict)
#== Predict the single value ==


#graph
plt.scatter(x_test, y_test, color = "Blue")
plt.plot(x_test, y_predict, color = "Red")
plt.show()


#Task 1 --

#===== MSE Start ===== 
MSE = mean_squared_error(y_test, y_predict)
print(f"This is MSE value: {MSE}")
#===== MSE end =====


#===== Absolute_error start =====
absolute_error = mean_absolute_error(y_test_value, y_predict_value)
print(f"This is Absolute error value: {absolute_error}")
#===== Absolute_error end =====


#===== Root square error start =====
root_square_error = np.sqrt(mean_squared_error(y_test, y_predict))
print(f'This is root square error value: {root_square_error}')




