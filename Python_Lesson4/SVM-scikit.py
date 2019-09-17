from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
import numpy as np

glass_data = pd.read_csv('glass.csv')

label_encoder = preprocessing.LabelEncoder()
data = glass_data
glass_data = label_encoder.fit_transform(glass_data.values[:, 9])

X = data.values[:, :4]
Y = glass_data



model = svm.SVC(gamma='scale', kernel='rbf')

kf = KFold(n_splits=15, shuffle=True,random_state=50)
accuracy = []

for train_index, test_index in kf.split(X):
   X_train, X_test = X[train_index], X[test_index]
   Y_train, Y_test = Y[train_index], Y[test_index]

   model.fit(X_train, Y_train)

   y_pred = model.predict(X_test)

   accuracy.append(metrics.accuracy_score(Y_test, y_pred))



print("Mean Accuracy Score: ", np.mean(accuracy))