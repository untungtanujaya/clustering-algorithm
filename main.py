#MAIN FUNGSI
#from agglomerative import agglomerative_scikit
from sklearn.cluster import AgglomerativeClustering
from agglomerative.agglomerative_implementation import AgglomerativeClusteringImp

from dbscan.dbscan_implementation import dbscan 

from sklearn.cluster import DBSCAN
import numpy as np
import csv

#LOAD DATA CSV menjadi array of dataset dan array of label
def load_csv(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        #MAKE DUMMY ARRAY
        X = np.zeros((1,4))
        label = np.array(None)
        Xtest = np.zeros((1,4))
        labeltest = np.array(None)
        for row in csv_reader:
            if line_count == 0:
                #NOT USING THE COLUMN NAMES
#                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                #ADD ELEMENT AND LABEL TO ARRAY
                if line_count % 50 > 44:
                    Xtest = np.append(Xtest, np.array([[float(row[0]),float(row[1]),float(row[2]),float(row[3])]]), axis=0)
                    labeltest = np.append(labeltest, row[4])
                else:
                    X = np.append(X, np.array([[float(row[0]),float(row[1]),float(row[2]),float(row[3])]]), axis=0)
                    label = np.append(label, row[4])
                line_count += 1
        #REMOVE DUMMY ELEMENT
        X = np.delete(X, 0, 0)
        label = np.delete(label, 0)
        Xtest = np.delete(Xtest, 0, 0)
        labeltest = np.delete(labeltest, 0)
    return X, label, Xtest, labeltest

#HANYA UNTUK DATA IRIS
def label_encode(label):
    labelmap = {}
    next_labelmap = 0
    ret_label = [0 for i in range(len(label))]
    for i in range(len(label)):
        if label[i] not in labelmap.keys():
            labelmap[label[i]] = next_labelmap
            next_labelmap += 1
        ret_label[i] = labelmap[label[i]]
    return ret_label

X, label, Xtest, labeltest = load_csv('iris.csv')
#print(f'{X}')
#print(f'{label}')
label = label_encode(label)
labeltest = label_encode(labeltest)
#print(f'{label}')
#print(f'{Xtest}')
#print(f'{labeltest}')

#Split data menjadi training dan test

#Ubah array of label menjadi numerik

#Main menu
print(f'###########################')
print(f'1. K-means')
print(f'2. Agglomerative')
print(f'3. DBScan')
print(f'###########################')
algo = input(f'Masukkan input: ')
#Pilihan K-means, Agglomerative, DBScan
if algo == '1':
    #K-means
    print(f'DO K-MEANS HERE')
        #Train and Test
elif algo == '2':
    #Agglomerative
    print(f'###########################')
    print(f'1. Single Linkage')
    print(f'2. Complete Linkage')
    print(f'3. Average Linkage')
    print(f'4. Average-Group Linkage')
    print(f'###########################')
    linkage = input("Masukkan Linkage: ")
    clustering = None
    #Pilih Linkage
    if linkage == '1':
        #Single
        clustering = AgglomerativeClusteringImp(n_clusters_=3, linkage='single').fit(X)
        print(f'Our Train label is: {clustering.labels_}')
        clustering_sklearn = AgglomerativeClustering(n_clusters=3, linkage='single').fit(X)
        print(f'Sklearn Train label is: {clustering_sklearn.labels_}')
            #Train and Test
    elif linkage == '2':
        #Complete
        clustering = AgglomerativeClusteringImp(n_clusters_=3, linkage='complete').fit(X)
        print(f'Our Train label is: {clustering.labels_}')
        clustering_sklearn = AgglomerativeClustering(n_clusters=3, linkage='complete').fit(X)
        print(f'Sklearn Train label is: {clustering_sklearn.labels_}')
            #Train and Test
    elif linkage == '3':
        #Average
        clustering = AgglomerativeClusteringImp(n_clusters_=3, linkage='average').fit(X)
        print(f'Our Train label is: {clustering.labels_}')
        clustering_sklearn = AgglomerativeClustering(n_clusters=3, linkage='average').fit(X)
        print(f'Sklearn Train label is: {clustering_sklearn.labels_}')
            #Train and Test
    elif linkage == '4':
        #Average-Group
        clustering = AgglomerativeClusteringImp(n_clusters_=3, linkage='average-group').fit(X)
        print(f'Our Train label is: {clustering.labels_}')
        #clustering_sklearn = AgglomerativeClustering(n_clusters=3, linkage='average-group').fit(X)
        #print(f'Sklearn Train label is: {clustering_sklearn.labels_}')
            #Train and Test
    accuracy = clustering.predict(Xtest, labeltest)
    print(f'Accuracy is: %.1f percent' %(accuracy))
elif algo == '3':    
    eps = 0.5
    min_p = 5

    # Train
    dbscan = dbscan(X)
    dbscan.fit(eps, min_p)

    print('Our Train Label Is:\n', dbscan.labels)
    print('Sklearn Train Label Is:\n', DBSCAN(eps=eps, min_samples=min_p).fit(X).labels_)

    # Test
    dbscan.predict(Xtest, labeltest)
else:
    print(f'EXITED')