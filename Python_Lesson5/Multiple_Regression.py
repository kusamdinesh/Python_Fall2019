import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#Loading the data

w = pd.read_csv('winequality-red.csv')

##Null values

n = pd.DataFrame(w.isnull().sum().sort_values(ascending=False))
n.columns = ['Null Count']
n.index.name = 'Feature'

print(n)



#Finding correlation with the target class

features = w.select_dtypes(include=[np.number])

corr = features.corr()
print(corr['quality'].sort_values(ascending=False), '\n')



#Dropping the columns having less correlation with target class

w=w.drop(columns=['residual sugar','free sulfur dioxide','pH'],axis=1)


print(w.columns)



#Splitting data

X = w.drop('quality', axis=1)

y = w['quality']

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=200)



#creation of regression model and training it

model=LinearRegression().fit(X_train,y_train)



#predicting the target

predict=model.predict(X_test)



#evaluation of model using metrics

mse = mean_squared_error(y_test, predict)

r2 = r2_score(y_test, predict)

print("Mean squared error :", mse)

print("R2 score : ", r2)