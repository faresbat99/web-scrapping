import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df = pd.read_csv("D:\data1.csv")
print(df.head())
#put points
plt.scatter(df.review, df.price)
plt.xlabel('review')
plt.ylabel('price')
plt.show()

scalar = MinMaxScaler()
scalar.fit_transform(df[['review', 'price']])
#get the best number of cluster
km = KMeans(n_clusters=5)
#get the points 
predicate = km.fit_predict(df[['review', 'price']])
df['cluster'] = predicate
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
df4 = df[df.cluster == 3]
df5 = df[df.cluster == 4]
plt.scatter(df1.review, df1.price, color="red", label="cluster1")
plt.scatter(df2.review, df2.price, color="green", label="cluster2")
plt.scatter(df3.review, df3.price, color="blue", label="cluster3")
plt.scatter(df4.review, df4.price, color="yellow", label="cluster4")
plt.scatter(df5.review, df5.price, color="aqua", label="cluster5")
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color="black", label="centroid")
plt.xlabel("review")
plt.ylabel("price")
plt.legend()
plt.show()

sse = []
for i in range(1, 26):
    km = KMeans(n_clusters=i)
    km.fit(df[['review', 'price']])
    sse.append(km.inertia_)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(range(1, 26), sse)
plt.show()
