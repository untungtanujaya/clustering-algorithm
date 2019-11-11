#MAIN FUNGSI
from agglomerative import agglomerative_scikit
from agglomerative import agglomerative_implementation

from dbscan.dbscan_implementation import dbscan 

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
print(f'{X}')
print(f'{label}')
label = label_encode(label)
print(f'{label}')

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
    #Pilih Linkage
        #Single
            #Train and Test
        #Complete
            #Train and Test
        #Average
            #Train and Test
        #Average-Group
            #Train and Test
elif algo == '3':    
    eps = 0.5
    min_p = 5
    filename = 'iris.csv'

    #Train and Test
    dbscan = dbscan(filename)
    labels = dbscan.fit(eps, min_p)
    print('from implementation\n', labels)

else:
    print(f'EXITED')