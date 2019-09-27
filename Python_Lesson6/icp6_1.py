
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('CC.csv')
dataset=dataset.apply(LabelEncoder().fit_transform)
nulls = pd.DataFrame(dataset.isnull().sum().
sort_values(ascending=False)[:25])
nulls.columns  = ['Null Count']
print(nulls)
dataset['CREDIT_LIMIT'].fillna((dataset['CREDIT_LIMIT'].mean()), inplace=True)
print(dataset["CREDIT_LIMIT"].isnull().any())
dataset['MINIMUM_PAYMENTS'].fillna((dataset['MINIMUM_PAYMENTS'].mean()), inplace=True)
print(dataset["MINIMUM_PAYMENTS"].isnull().any())

x = dataset.iloc[1:]
print(x)
nclusters = 3 # this is the k in kmeans
seed = 0

km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(x)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
print(y_cluster_kmeans)

wcss= []
for i in range(1,15):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,15),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
