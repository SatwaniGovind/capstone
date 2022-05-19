import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io
import sklearn
warnings.filterwarnings('ignore')
import pickle

from google.colab import files
uploaded = files.upload()

data = pd.read_csv(io.BytesIO(uploaded['Cardiology Admittance Dataset.csv']))
print("Cardiology Admittance Dataset. Size={}".format(data.shape))
data.tail()

data.corr()
sns.heatmap(data.corr(), annot = True)
plt.show()

X = data.iloc[:, [2,3,4,5,6,7,8]].values
Y = data.iloc[:, [9]].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)
# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("Y_train shape:", Y_train.shape)
# print("Y_test shape:", Y_test.shape)

from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)

from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dectree.fit(X_train, Y_train)

from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', max_features=3, oob_score=True, random_state = 42)
ranfor.fit(X_train, Y_train)

from sklearn.neighbors import KNeighborsClassifier as KNN
knn = KNN(n_neighbors = 3)
knn.fit(X_train, Y_train)

Y_pred_svc = svc.predict(X_test)
Y_pred_dectree = dectree.predict(X_test)
Y_pred_ranfor = ranfor.predict(X_test)
Y_pred_knn = knn.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_svc = accuracy_score(Y_test, Y_pred_svc)
accuracy_dectree = accuracy_score(Y_test, Y_pred_dectree)
accuracy_ranfor = accuracy_score(Y_test, Y_pred_ranfor)
accuracy_knn = accuracy_score(Y_test, Y_pred_knn)

a=accuracy_svc * 100
b=accuracy_dectree * 100
c=accuracy_ranfor * 100
d=accuracy_knn * 100
scores = [a,b,c,d]
algorithms = ["Support Vector Machine","Decision Tree","Random Forest","KNN"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")
sns.set(rc={'figure.figsize':(8,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
sns.barplot(algorithms,scores)

test = [70,1,2,0,0,1,150]
ranfor.predict([test])

test = [56,0,5,0,0,0,164]
ranfor.predict([test])

with open('pickled_data_file.pkl', 'wb') as fid:
  pickle.dump(ranfor,fid)

with open('pickled_data_file.pkl', 'rb') as fid:
  model1 = pickle.load(fid)
model1.predict([[70,1,2,0,0,1,150]])