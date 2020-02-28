import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('CC.csv')

#Handling NULL Values
nulls = pd.DataFrame(dataset.isnull().sum().sort_values(ascending=False)[:5])
nulls.columns  = ['Null Count']
nulls.index.name  = 'Feature'
print(nulls)

dataset['MINIMUM_PAYMENTS'].fillna(dataset['MINIMUM_PAYMENTS'].mean(), inplace=True)
dataset['CREDIT_LIMIT'].fillna(dataset['CREDIT_LIMIT'].mean(), inplace=True)

# nulls = pd.DataFrame(dataset.isnull().sum().sort_values(ascending=False)[:5])
# nulls.columns  = ['Null Count']
# nulls.index.name  = 'Feature'
# print(nulls)

x = dataset.iloc[:,1:-1]

#elbow method
wcss = []
for i in range(1,7):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1,7),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

km = KMeans(n_clusters=3)
km.fit(x)
y_cluster_kmeans= km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print(score)

scaler = StandardScaler()
scaler.fit(x)
x_scaler=scaler.transform(x)
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)

# #elbow method
# wcss = []
# for i in range(1,7):
#     kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
#     kmeans.fit(x_pca)
#     wcss.append(kmeans.inertia_)
# print(wcss)
# plt.plot(range(1,7),wcss)
# plt.title('the elbow method')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Wcss')
# plt.show()
#

km = KMeans(n_clusters=3)
km.fit(x_pca)
y_cluster_kmeans= km.predict(x_pca)
from sklearn import metrics
score = metrics.silhouette_score(x_pca, y_cluster_kmeans)
print(score)

print(y_cluster_kmeans)

plt.scatter(x_pca[:,0],x_pca[:,1],c=y_cluster_kmeans,cmap='rainbow')
plt.show()