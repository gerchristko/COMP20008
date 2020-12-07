import pandas as pd
import numpy as np
import csv
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier


# opening output csv
FILENAME1 = 'task2a.csv'
file1 = open(FILENAME1 ,'w+')
writer1 = csv.writer(file1)
writer1.writerow(['feature', 'median', 'mean', 'variance'])
# opening output csv


# opening input csv
life = pd.read_csv("life.csv", encoding='ISO-8859-1')
world = pd.read_csv("world.csv", encoding='ISO-8859-1')
world["Life expectancy at birth (years)"] = np.nan
# opening input csv


# merge dataframe
for i in life.index:
    exist = 0
    for j in world.index:
        if life["Country Code"][i] == world["Country Code"][j]:
            world["Life expectancy at birth (years)"][j] = life["Life expectancy at birth (years)"][i]
            break
# merge dataframe


# cleaning up the data      
world.replace("..", np.nan, inplace=True)
world.dropna(subset = ["Life expectancy at birth (years)"], inplace = True) 
world = world.sort_values(["Country Code"], ascending=True)
x_data = world.loc[:, ~world.columns.isin(["Life expectancy at birth (years)", "Country Name", "Country Code", "Time"])].astype(float)
y_data = world["Life expectancy at birth (years)"]
# cleaning up the data 


# split, impute, and scale data   
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 200)
columns = list(x_train)
for feature in columns:
    x_train[feature] = x_train[feature].astype(float)
    x_test[feature] = x_test[feature].astype(float)
    data1 = x_train[feature]
    data2 = x_test[feature]
    median = data1.median()
    mean = data1.mean()
    var = data1.var()
    writer1.writerow([feature, median, mean, var])
    data1.fillna(median, inplace=True)
    data2.fillna(median, inplace=True)
    x_train[feature] = data1
    x_test[feature] = data2
scaler = preprocessing.StandardScaler().fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
# split, impute, and scale data   


# DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 20, max_depth=3)
fitd = dt.fit(x_train, y_train)
predictdt = dt.predict(x_test)
print("Accuracy of decision tree:",round(accuracy_score(y_test, predictdt), 3))
# DecisionTreeClassifier


# 3-NN
knn3 = neighbors.KNeighborsClassifier(n_neighbors = 3)
fit3 = knn3.fit(x_train, y_train)
predict3 = knn3.predict(x_test)
print("Accuracy of k-nn (k=3):",round(accuracy_score(y_test, predict3), 3))
# 3-NN


# 7-NN
knn7 = neighbors.KNeighborsClassifier(n_neighbors = 7)
fit7 = knn7.fit(x_train, y_train)
predict7 = knn7.predict(x_test)
print("Accuracy of k-nn (k=7):",round(accuracy_score(y_test, predict7), 3))
# 7-NN


file1.close()
