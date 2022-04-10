import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('/Users/anastasiacyntia/Library/Containers/com.microsoft.Excel/Data/Downloads/covid19indonesia.csv')
dataset.head()
dataset.info()

sns.heatmap(dataset.isnull(),yticklabels=False, cbar=False, cmap="viridis")
dataset.corr()
sns.heatmap(dataset.corr())
plt.show()


datacluster = dataset.iloc[:,1:4]
datacluster.head()
sns.scatterplot(x="terkonfirmasi", y="meninggal", data=datacluster, s=100, color="red", alpha=0.5)
plt.show()
sns.scatterplot(x="terkonfirmasi", y="sembuh", data=datacluster, s=100, color="red", alpha=0.5)
plt.show()

dataArray = np.array(datacluster)
print(dataArray)

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
data_scale = min_max_scaler.fit_transform(dataArray)

# from sklearn import cluster
# K=range(1, 11)
# wcss = []
# for k in K:
    # kmeans= cluster.KMeans(n_clusters=k,init="k-means++")
    # kmeans= kmeans.fit(data_scale)
    # wcss_iter = kmeans.inertia_
    # wcss.append(wcss_iter)

# mycenters = pd.DataFrame({'Cluster' : K, 'WCSS' : wcss})
# mycenters

# sns.scatterplot(x = 'Cluster', y = 'WCSS', data =mycenters, marker="+")
# plt.show()

# The Elbow Method
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10 , random_state = 0)
    kmeans.fit(data_scale)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Clustering with KMeans n=4
dataKmeans = KMeans(n_clusters = 4, init = 'k-means++', n_init=1, random_state=0)
dataKmeans.fit(data_scale)
label = dataKmeans.predict(data_scale)

print(dataKmeans.cluster_centers_)
print(dataKmeans.labels_)

dataset["Cluster"] = dataKmeans.labels_
dataset.head()

center = dataKmeans.cluster_centers_
 # CONFIRMED COVID 19  With Patient Recorverd
plt.scatter(data_scale[label == 0, 0], data_scale[label == 0, 1], s = 100, c = 'red', label = '0.Cluster A')
plt.scatter(data_scale[label == 1, 0], data_scale[label == 1, 1], s = 100, c = 'blue', label = '1.Cluster B')
plt.scatter(data_scale[label == 2, 0], data_scale[label == 2, 1], s = 100, c = 'yellow', label = '2.Cluster C')
plt.scatter(data_scale[label == 3, 0], data_scale[label == 3, 1], s = 100, c = 'green', label = '3.Cluster D')
plt.scatter(center[:,2], center[:,0], marker='+', c='black',s=200, alpha=0.5, label= 'Centroids')


 


 # CONFIRMED COVID 19  With Death Cases
plt.scatter(data_scale[label == 0, 0], data_scale[label == 0, 2], s = 100, c = 'red', label = '0.Cluster A')
plt.scatter(data_scale[label == 1, 0], data_scale[label == 1, 2], s = 100, c = 'blue', label = '1.Cluster B')
plt.scatter(data_scale[label == 2, 0], data_scale[label == 2, 2], s = 100, c = 'yellow', label = '2.Cluster C')
plt.scatter(data_scale[label == 3, 0], data_scale[label == 3, 2], s = 100, c = 'green', label = '3.Cluster D')
plt.scatter(center[:,2], center[:,0], marker='+', c='black',s=200, alpha=0.5, label= 'Centroids')


plt.legend()
plt.title ('Clustering of COVID 19 in Indonesia CONFIRMED COVID 19  With Death Cases')
plt.show()



