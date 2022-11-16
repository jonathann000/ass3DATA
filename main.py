
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

df = pd.read_csv('assignment3-data-1.csv')

# make dataset for clustering with phi and psi
df = df[['phi', 'psi']]
df = df.dropna()

# heatmap with x and y as phi and psi with plt
plt.hist2d(df['phi'], df['psi'], bins=500)
plt.show()

# Perform k-means clustering on df with 3 clusters
# Clustering with more than 3 clusters doesn't seem that accurate using k-means.
kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)
kmeans.fit(df)
y_kmeans = kmeans.predict(df)

# Plot the cluster centers and the data points on a 2D plane
plt.scatter(df['phi'], df['psi'], c=y_kmeans, s=50, cmap='Oranges', edgecolor='black', linewidths=0.5)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='limegreen', s=200, alpha=1, marker='X', linewidths=1)
plt.show()

# Validating clustering with silhouette score
#print(silhouette_score(df, y_kmeans))


# remove portion of random points from df
dfSample = df.sample(frac=0.1, random_state=0)


# create co-occurrence matrix
coOccurrenceMatrix = np.zeros((3, 3))
for i in range(len(dfSample)):
    coOccurrenceMatrix[y_kmeans[i], y_kmeans[i+1]] += 1

# plot co-occurrence matrix
plt.imshow(coOccurrenceMatrix, cmap='Oranges')
plt.show()

kmeansSample = KMeans(n_clusters=3)
kmeansSample.fit(dfSample)
y_kmeansSample = kmeansSample.predict(dfSample)

# Plot the cluster centers and the data points on a 2D plane
plt.scatter(dfSample['phi'], dfSample['psi'], c=y_kmeansSample, s=50, cmap='Oranges', edgecolor='black', linewidths=0.5)
centers = kmeansSample.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='limegreen', s=200, alpha=1, marker='X', linewidths=1)
plt.show()

#print(silhouette_score(dfSample, y_kmeansSample))

# Can you change the data to get better results (or the same results in a simpler
# way)? (Hint: since both phi and psi are periodic attributes, you can think of
# shifting/translating them by some value and then use the modulo operation.)

# Use DBSCAN to cluster the data. How does it compare to k-means?

# DBSCAN
# Compute DBSCAN #eps=20 minS=50
db = DBSCAN(eps=20, min_samples=50).fit(df)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
#print("Silhouette Coefficient: %0.3f"
#        % metrics.silhouette_score(df, labels))

# Plot result
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = df[class_member_mask
            & core_samples_mask]
    plt.plot(xy['phi'], xy['psi'], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=14)

    xy = df[class_member_mask
            & ~core_samples_mask]
    plt.plot(xy['phi'], xy['psi'], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

# Create a new dataset keeping only phi, psi and chain
dfclean = pd.read_csv('assignment3-data-1.csv')
df2 = dfclean[['phi', 'psi', 'chain']]
df2 = df2.dropna()

# Use DBScan and ignore chain
db = DBSCAN(eps=20, min_samples=50).fit(df2[['phi', 'psi']])
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# create dataset with only noise points
dfNoise = df2[labels == -1]

# Plot bar chart of number of noise points per chain
noise_plot = dfNoise['chain'].value_counts().plot(kind='bar', color='orange')
noise_plot.set_facecolor('black')
plt.show()

# Create new dataset with only residue name PRO
dfPro = dfclean[dfclean['residue name'] == 'PRO']

# Use DBScan and to cluster the data with residue type PRO
db = DBSCAN(eps=20, min_samples=50).fit(dfPro[['phi', 'psi']])
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
#print("Silhouette Coefficient: %0.3f"
#        % metrics.silhouette_score(df, labels))

# Plot result
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = dfPro[class_member_mask
            & core_samples_mask]
    plt.plot(xy['phi'], xy['psi'], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=14)

    xy = dfPro[class_member_mask
            & ~core_samples_mask]
    plt.plot(xy['phi'], xy['psi'], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=6)


plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

# Create new dataset with only residue name GLY

dfGly = dfclean[dfclean['residue name'] == 'GLY']

# Use DBScan and to cluster the data with residue type GLY
db = DBSCAN(eps=20, min_samples=50).fit(dfGly[['phi', 'psi']])
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
#print("Silhouette Coefficient: %0.3f"
#        % metrics.silhouette_score(df, labels))

# Plot result
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = dfGly[class_member_mask
            & core_samples_mask]
    plt.plot(xy['phi'], xy['psi'], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=14)

    xy = dfGly[class_member_mask
            & ~core_samples_mask]
    plt.plot(xy['phi'], xy['psi'], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

# 4 hours later...












