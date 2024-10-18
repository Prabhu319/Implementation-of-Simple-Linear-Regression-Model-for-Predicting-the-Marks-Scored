# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: prabanjan m
RegisterNumber: 24900428 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:
![ex21](https://github.com/user-attachments/assets/de507f5e-2c50-40c9-8e88-b586902aa514)
![ex22](https://github.com/user-attachments/assets/f574b736-b193-4d1a-a1ac-8ce85c8fd330)
MSE =  25.463280738222547
MAE =  4.691397441397438
RMSE =  5.046115410711743



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
•	The first part is importing the necessary libraries for data manipulation, visualization, and machine learning. Pandas is used for reading and processing the data, numpy is used for numerical operations, matplotlib is used for plotting graphs, and sklearn is used for importing the metrics and the model.
•	The second part is loading the dataset from a CSV file. The dataset contains two columns: Hours and Scores. Hours is the number of hours studied by a student, and Scores is the percentage of marks obtained by the student. The code uses the pandas function read_csv to load the data into a dataframe, which is a tabular data structure. The code then prints the first and last five rows of the data using the head and tail methods, and also prints some basic information and statistics about the data using the info and describe methods.
•	The third part is plotting the dataset using matplotlib. The code uses the scatter function to plot the Hours and Scores columns as points on a graph, and labels the axes and the title. The graph shows a positive linear relationship between the two variables, meaning that as the hours of study increase, the scores also increase.
•	The fourth part is splitting the dataset into features and target, and then into training and testing sets. The features are the independent variables that are used to predict the target, which is the dependent variable. In this case, the feature is Hours and the target is Scores. The code uses the numpy function iloc to select the columns by their index, and assigns them to X and y respectively. The code then uses the sklearn function train_test_split to split the data into two subsets: one for training the model and one for testing the model. The code specifies that the test size is 1/3, meaning that 33% of the data is used for testing and the rest for training. The code also sets a random state of 0, which is a seed for the random number generator that ensures the same split every time the code is run.
•	The fifth part is creating and fitting the linear regression model. The code uses the sklearn class LinearRegression to create an instance of the model, and then calls the fit method to train the model on the training data. The model learns the relationship between the features and the target by finding the best line that minimizes the error between the actual and predicted values.
•	The sixth part is making predictions on the testing set. The code uses the predict method of the model to generate the predicted values for the test features, and assigns them to y_pred. The code then prints the predicted and actual values for comparison.
•	The seventh part is evaluating the model performance. The code uses the sklearn functions mean_squared_error, mean_absolute_error, and np.sqrt to calculate the mean squared error (MSE), the mean absolute error (MAE), and the root mean squared error (RMSE) respectively. These are common metrics for measuring the accuracy of regression models. They represent the average difference between the actual and predicted values, squared or not. The lower the values, the better the model.
•	The eighth part is plotting the regression line and the testing data. The code uses the matplotlib function scatter to plot the test features and target as points on a graph, and then uses the plot function to plot the test features and the predicted values as a line on the same graph. The code labels the axes and the title, and uses different colors for the points and the line. The graph shows how well the model fits the data, and how close the predicted values are to the actual values.
I hope this explanation helps you understand the code better. If you have any questions or feedback, please let me know. 😊


