#from sklearn.metrics import pairwise_distances
#from sklearn.metrics.pairwise import euclidean_distances
#from sklearn.metrics.cluster import homogeneity_score
#from sklearn.metrics.cluster import completeness_score
#from sklearn.metrics.cluster import v_measure_score
import math

class AgglomerativeClusteringImp:

    n_clusters_ =  2
    labels_ = []
    linkage_ = 'average-group'
    training_data = None

    def __init__(self, linkage = 'average-group', n_clusters_ = 2):
        self.linkage = linkage 
        self.n_clusters_ = n_clusters_ 

    def fit(self, X):
        #Banyak Cluster awal
        n_clusters = len(X)

        #Clone Data Training
        self.training_data = [[X[i][j] for j in range(len(X[i]))] for i in range(len(X))]

        #Pembuatan Matriks Jarak
        distance_matrix = [[0 for i in range(len(X))] for j in range(len(X))]
        for i in range(len(X)):
            for j in range(i):
                for k in range(len(X[i])):
                    distance_matrix[i][j] += (X[i][k] - X[j][k])**2
                    distance_matrix[j][i] += (X[i][k] - X[j][k])**2
                distance_matrix[i][j] = math.sqrt(distance_matrix[i][j])
                distance_matrix[j][i] = math.sqrt(distance_matrix[j][i])

        if self.linkage == 'ward':
            for i in range(len(X)):
                for j in range(len(X)):
                    distance_matrix[i][j] *= distance_matrix[i][j]

        #Inisialisasi Label
        labels = [i for i in range(n_clusters)]

        #Centroid setiap Cluster (Untuk Average-Group)
        cluster_centroid = [[X[i][j] for j in range(len(X[i]))] for i in range(len(X))]

        #Size setiap Cluster (Untuk Average dan Average-Group)
        cluster_size = [1 for i in range(n_clusters)]

        while n_clusters > self.n_clusters_:
            #Inisialisasi label yang akan dimerge
            label_min1 = None
            label_min2 = None
            value_min = None
            for i in range(len(X)):
                for j in range(i):
                    if i != j and labels[i] == i and labels[j] == j:
                        if value_min == None or distance_matrix[i][j] < value_min:
                            value_min = distance_matrix[i][j]
                            label_min1 = i
                            label_min2 = j
            #Menggabungkan Cluster. Kalau parent berubah, semua cluster sama dengan parent tersebut diubah
            for i in range(len(X)):
                if labels[i] == label_min2:
                    labels[i] = label_min1
            if self.linkage == 'single':
                for i in range(len(X)):
                    if labels[i] == i and i != label_min2:
                        distance_matrix[i][label_min1] = min(distance_matrix[i][label_min1], distance_matrix[i][label_min2])
                        distance_matrix[label_min1][i] = min(distance_matrix[label_min1][i], distance_matrix[label_min2][i])
                #Perhitungan Jarak baru
                pass
            elif self.linkage == 'complete':
                for i in range(len(X)):
                    if labels[i] == i and i != label_min2:
                        distance_matrix[i][label_min1] = max(distance_matrix[i][label_min1], distance_matrix[i][label_min2])
                        distance_matrix[label_min1][i] = max(distance_matrix[label_min1][i], distance_matrix[label_min2][i])
                pass
            elif self.linkage == 'average':
                for i in range(len(X)):
                    if labels[i] == i and i != label_min2:
                        distance_matrix[i][label_min1] = (cluster_size[label_min1]*distance_matrix[i][label_min1] + cluster_size[label_min2]*distance_matrix[i][label_min2])/(cluster_size[label_min1]+cluster_size[label_min2])
                        distance_matrix[label_min1][i] = (cluster_size[label_min1]*distance_matrix[label_min1][i] + cluster_size[label_min2]*distance_matrix[label_min2][i])/(cluster_size[label_min1]+cluster_size[label_min2])
                cluster_size[label_min1] += cluster_size[label_min2]
                cluster_size[label_min2] = 0
                pass
            elif self.linkage == 'ward':
                ni = cluster_size[label_min1]
                nj = cluster_size[label_min2]
                for i in range(len(X)):
                    if labels[i] == i and i != label_min1 and i != label_min2:
                        nk = cluster_size[i]
                        sum = ni + nj + nk
                        distance_matrix[i][label_min1] = (ni + nk)*(distance_matrix[i][label_min1])/sum + (nj + nk)*(distance_matrix[i][label_min2])/sum - (nk)*(distance_matrix[label_min2][label_min1])/sum
                        distance_matrix[label_min1][i] = (ni + nk)*(distance_matrix[label_min1][i])/sum + (nj + nk)*(distance_matrix[label_min2][i])/sum - (nk)*(distance_matrix[label_min1][label_min2])/sum
                cluster_size[label_min1] += cluster_size[label_min2]
                cluster_size[label_min2] = 0
                pass
            else:
                #Default Average-Group
                centroid_value = [0 for i in range(len(cluster_centroid[label_min1]))]
                for i in range(len(centroid_value)):
                    centroid_value[i] = cluster_centroid[label_min1][i]*cluster_size[label_min1] + cluster_centroid[label_min2][i]*cluster_size[label_min2] 
                    centroid_value[i] /= cluster_size[label_min1] + cluster_size[label_min2]
                cluster_centroid[label_min1] = centroid_value
                for i in range(len(X)):
                    if labels[i] == i and i != label_min2:
                        tmp = 0
                        for k in range(len(cluster_centroid[label_min1])):
                            tmp += (cluster_centroid[label_min1][k] - cluster_centroid[i][k])**2
                        tmp = math.sqrt(tmp)
                        distance_matrix[i][label_min1] = tmp
                        distance_matrix[label_min1][i] = tmp
                cluster_size[label_min1] += cluster_size[label_min2]
                cluster_size[label_min2] = 0
                cluster_centroid[label_min2] = None
                pass
            #Null kan nilai cluster yang dimerge ke cluster lain
            for i in range(len(X)):
                distance_matrix[i][label_min2] = None
                distance_matrix[label_min2][i] = None
            n_clusters = n_clusters - 1
        self.labels_ = labels
#        self.labels_ = self.label_encode(labels)
        return self
'''   
    #Hanya dipakai di dalam class
    def label_encode(self, label):
        labelmap = {}
        next_labelmap = 0
        ret_label = [0 for i in range(len(label))]
        for i in range(len(label)):
            if label[i] not in labelmap.keys():
                labelmap[label[i]] = next_labelmap
                next_labelmap += 1
            ret_label[i] = labelmap[label[i]]
        return ret_label

    def predict(self, X):
        ret = [None for i in range(len(X))]
        
        distance_matrix = 3
        
        return ret

#READ CSV to datasets and labels
import numpy as np
import csv
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
            X = np.append(X, np.array([[float(row[0]),float(row[1]),float(row[2]),float(row[3])]]), axis=0)
            label = np.append(label, row[4])
            line_count += 1
    #REMOVE DUMMY ELEMENT
    X = np.delete(X, 0, 0)
    label = np.delete(label, 0)

    clustering = AgglomerativeClusteringImp(linkage='single', n_clusters_=3).fit(X)
    print(f'Labels = {clustering.labels_}')
    clusteringlabel_nums = AgglomerativeClusteringImp().label_encode(clustering.labels_)
    print(f'Cluster Labels = {clusteringlabel_nums}')

    labelarray = AgglomerativeClusteringImp().label_encode(label)

    print(f'Labels unique : {labelarray}')

    #SCORING
    print("%.6f" % v_measure_score(labelarray, clustering.labels_))
    print("%.6f" % completeness_score(labelarray, clustering.labels_))
    print("%.6f" % homogeneity_score(labelarray, clustering.labels_))

X = [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.5], [0.4, 0.5, 0.6, 0.7], [0.4, 0.5, 0.6, 0.8], [0.8, 0.9, 1.0, 1.1]]
clusters = AgglomerativeClusteringImp(linkage='average-group', n_clusters_ = 3).fit(X)
print(f'Labels = {clusters.labels_}')
'''