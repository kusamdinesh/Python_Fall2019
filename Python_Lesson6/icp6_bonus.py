
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.decomposition import PCA
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


x = dataset.iloc[:,[0, 1, 2, 3, 4, 5, 12]]
print(x)



#Standardize features by removing the mean and scaling to unit variance.
scaler = preprocessing.StandardScaler()
#Compute the mean and std to be used for later scaling.
scaler.fit(x)
#Perform standardization by centering and scaling
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)


nclusters = 3 # this is the k in kmeans
seed = 0

km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X_scaled)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_scaled)
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

#pca
ndimensions = 2

pca = PCA(ndimensions)
X_pca = pca.fit_transform(X_scaled)

nclusters = 3 # this is the k in kmeans
seed = 0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X_pca)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_pca)
print(y_cluster_kmeans)
score = metrics.silhouette_score(X_pca, y_cluster_kmeans)
print("sil score after pca",score)