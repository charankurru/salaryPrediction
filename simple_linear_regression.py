# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
praveen = LinearRegression()
praveen.fit(X_train, y_train)

# Predicting the Test set results
y_pred = praveen.predict(X_test)

x_pred = praveen.predict(X_train)

char_pred  = praveen.predict([[2]])
# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, praveen.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

from sklearn.externals import joblib 

# Save the model as a pickle in a file 
joblib.dump(praveen, 'linearRegression.pkl') 

# Load the model from the file 
#knn_from_joblib = joblib.load('filename.pkl') 

# Use the loaded model to make predictions 
#knn_from_joblib.predict(X_test) 
