#MAIN FUNGSI
#from agglomerative import agglomerative_scikit
from agglomerative import agglomerative_implementation
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
        for row in csv_reader:
            if line_count == 0:
                #NOT USING THE COLUMN NAMES
#                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                #ADD ELEMENT AND LABEL TO ARRAY
                X = np.append(X, np.array([[float(row[0]),float(row[1]),float(row[2]),float(row[3])]]), axis=0)
                label = np.append(label, row[4])
                line_count += 1
        #REMOVE DUMMY ELEMENT
        X = np.delete(X, 0, 0)
        label = np.delete(label, 0)
    return X, label

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

X, label = load_csv('iris.csv')
#print(f'{X}')
#print(f'{label}')
label = label_encode(label)
#print(f'{label}')
#clustering = agglomerative_implementation.AgglomerativeClusteringImp(linkage='average-group', n_clusters_=3).fit(X)
#print(f'{clustering.labels_}')

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
    print(f'DO AGGLOMERATIVE HERE')
    print(f'###########################')
    print(f'1. Single Linkage')
    print(f'2. Complete Linkage')
    print(f'3. Average Linkage')
    print(f'4. Average-Group Linkage')
    print(f'###########################')
    linkage = input("Masukkan Linkage: ")
    #Pilih Linkage
    if linkage == '1':
        #Single
        print(f'DO SINGLE LINKAGE HERE')
        clustering = agglomerative_implementation.AgglomerativeClusteringImp(n_clusters_=3, linkage='single').fit(X)
        print(f'{label_encode(clustering.labels_)}')
            #Train and Test
    elif linkage == '2':
        #Complete
        print(f'DO COMPLETE LINKAGE HERE')
        clustering = agglomerative_implementation.AgglomerativeClusteringImp(n_clusters_=3, linkage='complete').fit(X)
        print(f'{label_encode(clustering.labels_)}')
            #Train and Test
    elif linkage == '3':
        #Average
        print(f'DO AVERAGE LINKAGE HERE')
        clustering = agglomerative_implementation.AgglomerativeClusteringImp(n_clusters_=3, linkage='average').fit(X)
        print(f'{label_encode(clustering.labels_)}')
            #Train and Test
    elif linkage == '4':
        #Average-Group
        print(f'DO AVERAGE-GROUP LINKAGE HERE')
        clustering = agglomerative_implementation.AgglomerativeClusteringImp(n_clusters_=3, linkage='average-group').fit(X)
        print(f'{label_encode(clustering.labels_)}')
            #Train and Test
    else:
        print(f'EXITED')
elif algo == '3':    
    #DBScan
    print(f'DO DBSCAN HERE')
        #Train and Test
else:
    print(f'EXITED')