import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, mutual_info_classif

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
world = world.reset_index()
world.drop(['index', 'Country Name', 'Time', 'Country Code'], axis=1, inplace = True)
nums = world.loc[:, ~world.columns.isin(["Life expectancy at birth (years)", "Country Name", "Country Code", "Time"])].astype(float)
# cleaning up the data 


# median imputation
x_data = world.loc[:, ~world.columns.isin(["Life expectancy at birth (years)", "Country Name", "Country Code", "Time"])].astype(float)
y_data = world["Life expectancy at birth (years)"]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state = 200)
columns = list(nums)
pca = world.copy()
for feature in columns:
    world[feature] = world[feature].astype(float)
    x_train[feature] = x_train[feature].astype(float)
    x_test[feature] = x_test[feature].astype(float)
    pca[feature] = pca[feature].astype(float)
    data1 = world[feature]
    data2 = x_train[feature]
    data3 = x_test[feature]
    data4 = pca[feature]
    median1 = data1.median()
    median2 = data2.median()
    data1.fillna(median2, inplace = True)
    data2.fillna(median2, inplace = True)
    data3.fillna(median2, inplace = True)
    data4.fillna(median1, inplace = True)
    world[feature] = data1
    x_train[feature] = data2
    x_test[feature] = data3
    pca[feature] = data4
fff = world.copy()
# median imputation


# k-means cluster
col = list(world)
wcss = []
df = world.loc[:, world.columns != "Life expectancy at birth (years)"]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state = 200)
    fitk = kmeans.fit(df)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig("task2bgraph1.png")
plt.show()
kmeans = KMeans(n_clusters = 2, random_state = 200)
fitk = kmeans.fit(df)
predict = kmeans.predict(df)
world['clusterlabel'] = predict
# k-means cluster


# creating term pairs
for feature in columns:
    index = world.columns.get_loc(feature)
    j = index + 1
    while (j < len(col) - 1):
        products = []
        for i in world.index:
            value = world[feature][i] * world.iloc[ : , j][i]
            products.append(value)
        world[str(index) + "*" + str(j)] = products
        j += 1
# creating term pairs


# FE
selector = SelectKBest(score_func = mutual_info_classif, k = 4).fit_transform(x_train, y_train)
best4 = np.transpose(selector[:5]).astype(float)
features = list(x_train)
predictor = []
for i in features:
    for j in best4:
        equals = 0
        for n in range(5):
            if x_train[i].iloc[n] == j[n]:
                equals += 1
        if equals == 5:
            predictor.append(i)
x_train_4 = x_train[predictor]
x_test_4 = x_test[predictor]
scaler = preprocessing.StandardScaler().fit(x_train_4)
x_train_4 = scaler.transform(x_train_4)
x_test_4 = scaler.transform(x_test_4)
knnfe = neighbors.KNeighborsClassifier(n_neighbors = 3)
fitfe = knnfe.fit(x_train_4, y_train)
predictfe = knnfe.predict(x_test_4)
print("Accuracy of feature engineering:",round(accuracy_score(y_test, predictfe), 3))
# FE


# PCA
covariance = pca.cov()
matrix = covariance.to_numpy()
eigenvalues, eigenvectors = np.linalg.eig(matrix)
featurevector = eigenvectors[0:4]
oridataset = pca.loc[:, ~pca.columns.isin(["Life expectancy at birth (years)", "Country Name", "Country Code", "Time"])].astype(float)
orimatrix = oridataset.to_numpy()
finaldataset = np.matmul(featurevector, orimatrix.transpose())
principles = pd.DataFrame()
for i in range(len(finaldataset)):
    principles[i] = finaldataset[i]
principles["Life expectancy at birth (years)"] = world["Life expectancy at birth (years)"]
x_data = principles.loc[:, ~principles.columns.isin(["Life expectancy at birth (years)"])].astype(float)
y_data = principles["Life expectancy at birth (years)"]
x_train_p, x_test_p, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state = 200)
scaler = preprocessing.StandardScaler().fit(x_train_p)
x_train_p = scaler.transform(x_train_p)
x_test_p = scaler.transform(x_test_p)
knnpc = neighbors.KNeighborsClassifier(n_neighbors = 3)
fitpc = knnpc.fit(x_train_p, y_train)
predictpc = knnpc.predict(x_test_p)
print("Accuracy of PCA:",round(accuracy_score(y_test, predictpc), 3))
# PCA


# FFF
x_train_f = x_train[["Access to electricity (% of population) [EG.ELC.ACCS.ZS]", 
                     "Adjusted net national income per capita (current US$) [NY.ADJ.NNTY.PC.CD]", 
                     "Age dependency ratio (% of working-age population) [SP.POP.DPND]", 
                     "Cause of death, by communicable diseases and maternal, prenatal and nutrition conditions (% of total) [SH.DTH.COMM.ZS]"]]
x_test_f = x_test[["Access to electricity (% of population) [EG.ELC.ACCS.ZS]", 
                     "Adjusted net national income per capita (current US$) [NY.ADJ.NNTY.PC.CD]", 
                     "Age dependency ratio (% of working-age population) [SP.POP.DPND]", 
                     "Cause of death, by communicable diseases and maternal, prenatal and nutrition conditions (% of total) [SH.DTH.COMM.ZS]"]]
scaler = preprocessing.StandardScaler().fit(x_train_f)
x_train_f = scaler.transform(x_train_f)
x_test_f = scaler.transform(x_test_f)
knnff = neighbors.KNeighborsClassifier(n_neighbors = 3)
fitff = knnff.fit(x_train_f, y_train)
predictff = knnff.predict(x_test_f)
print("Accuracy of first four features:",round(accuracy_score(y_test, predictff), 3))
# FFF


