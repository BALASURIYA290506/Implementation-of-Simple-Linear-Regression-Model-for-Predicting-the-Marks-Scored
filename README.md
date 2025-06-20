# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: Balasuriya M

RegisterNumber: 212224240021
```python
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv('student_scores.csv')

print(df)

df.head(0)

df.tail(0)

x = df.iloc[:,:-1].values

print(x)

y = df.iloc[:,1].values

print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print(y_pred)

print(y_test)

mae = mean_absolute_error(y_test,y_pred)

print("MAE: ",mae)

mse = mean_squared_error(y_test,y_pred)

print("MSE: ",mse)

rmse = np.sqrt(mse)

print("RMSE: ",rmse)

plt.scatter(x_train,y_train)

plt.plot(x_train,regressor.predict(x_train) , color ='blue')

plt.title("Hours vs Scores(training set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()

plt.scatter(x_test,y_test)

plt.plot(x_test,regressor.predict(x_test),color = 'black')

plt.title("Hours vs Scores(testing set)")

plt.xlabel("Hours")

plt.ylabel("Scores")

plt.show()
```

## Output:
![Screenshot 2025-04-19 222506](https://github.com/user-attachments/assets/9a7ce484-f047-4077-96ac-c5e67d5a0a7d)
![Screenshot 2025-04-19 222754](https://github.com/user-attachments/assets/b7aae0d4-c90d-4ef8-ac13-f7a9ae57236a)
![Screenshot 2025-04-19 222833](https://github.com/user-attachments/assets/761b1107-c7e5-42df-bc03-1cb3ac4eaf51)
![Screenshot 2025-04-19 222922](https://github.com/user-attachments/assets/c509da74-f45e-430a-b9ae-4696ddf15e3d)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
