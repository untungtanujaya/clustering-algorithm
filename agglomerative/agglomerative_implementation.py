from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

class AgglomerativeClusteringImp:

    n_clusters_ =  2
    labels_ = []
    linkage_ = 'ward'

    def __init__(self, linkage = 'ward', n_clusters_ = 2):
        self.linkage = linkage 
        self.n_clusters_ = n_clusters_ 

    def fit(self, X):
        #Banyak Cluster awal
        n_clusters = len(X)
        print(f'Number of Elements: {n_clusters}')
        #Pembuatan Matriks Jarak
        distance_matrix = pairwise_distances(X=X, metric='euclidean')

        #Inisialisasi Label
        labels = [i for i in range(n_clusters)]

        #Centroid setiap Cluster (Untuk Average)
        cluster_centroid = [[X[i][j] for j in range(len(X[i]))] for i in range(len(X))]

        #Size setiap Cluster (Untuk Average dan Ward)
        cluster_size = [1 for i in range(n_clusters)]

        while n_clusters > self.n_clusters_:
            print(f'Distance Matrix :{distance_matrix}')
            #Inisialisasi label yang akan dimerge
            label_min1 = None
            label_min2 = None
            value_min = None
            for i in range(len(X)):
                for j in range(i):
                    if i != j and labels[i] == i and labels[j] == j:
                        if value_min == None or distance_matrix[i][j] < value_min:
                            value_min = distance_matrix[i][j]
                            print(f'Value_min = {value_min}')
                            label_min1 = i
                            label_min2 = j
            print(f'Label Minimums: {label_min1}, {label_min2}')
            #Menggabungkan Cluster
            for i in range(len(X)):
                if labels[i] == label_min2:
                    labels[i] = label_min1
            if self.linkage == 'single':
                print(f'SINGLE IS BEING DONE')
                for i in range(len(X)):
                    if i != label_min2:
                        distance_matrix[i][label_min1] = min(distance_matrix[i][label_min1], distance_matrix[i][label_min2])
                        distance_matrix[label_min1][i] = min(distance_matrix[label_min1][i], distance_matrix[label_min2][i])
                #Perhitungan Jarak baru
                pass
            elif self.linkage == 'complete':
                print(f'COMPLETE IS BEING DONE')
                for i in range(len(X)):
                    if i != label_min2:
                        distance_matrix[i][label_min1] = max(distance_matrix[i][label_min1], distance_matrix[i][label_min2])
                        distance_matrix[label_min1][i] = max(distance_matrix[label_min1][i], distance_matrix[label_min2][i])
                pass
            elif self.linkage == 'average':
                print(f'AVERAGE IS BEING DONE')
                print(f'Cluster Centroid')
                print(f'{cluster_centroid}')
                centroid_value = [0 for i in range(len(cluster_centroid[label_min1]))]
                for i in range(len(centroid_value)):
                    centroid_value[i] = cluster_centroid[label_min1][i]*cluster_size[label_min1] + cluster_centroid[label_min2][i]*cluster_size[label_min2] 
                    centroid_value[i] /= cluster_size[label_min1] + cluster_size[label_min2]
                cluster_centroid[label_min1] = centroid_value
                for i in range(len(X)):
                    if labels[i] == i and i != label_min2:
                        distance_matrix[i][label_min1] = euclidean_distances([cluster_centroid[label_min1]], [cluster_centroid[i]])[0][0]
                cluster_size[label_min1] += cluster_size[label_min2]
                cluster_size[label_min2] = 0
                cluster_centroid[label_min2] = None
                pass
            else:
                #Default Ward
                print(f'WARD IS BEING DONE')

                pass
            #Null kan nilai cluster yang dimerge ke cluster lain
            for i in range(len(X)):
                distance_matrix[i][label_min2] = None
                distance_matrix[label_min2][i] = None
            n_clusters = n_clusters - 1
        self.labels_ = labels
        return self

import numpy as np
'''
import csv

#READ CSV
with open('iris.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    line_count = 0
    #MAKE DUMMY ARRAY
    X = np.zeros((1,4))
    label = np.array(None)
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            #ADD ELEMENT AND LABEL TO ARRAY
            X = np.append(X, np.array([[row[0],row[1],row[2],row[3]]]), axis=0)
            label = np.append(label, row[4])
            line_count += 1
    #REMOVE DUMMY ELEMENT
    X = np.delete(X, 0, 0)
    label = np.delete(label, 0)

    clustering = AgglomerativeClustering(linkage='single', n_clusters=3).fit(X)
'''
X = [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.5], [0.4, 0.5, 0.6, 0.7], [0.4, 0.5, 0.6, 0.8], [0.8, 0.9, 1.0, 1.1]]
clusters = AgglomerativeClusteringImp(linkage='average', n_clusters_ = 3).fit(X)
print(f'Labels = {clusters.labels_}')