from sklearn.cluster import AgglomerativeClustering
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
            X = np.append(X, np.array([[row[0],row[1],row[2],row[3]]]), axis=0)
            label = np.append(label, row[4])
            line_count += 1
    #REMOVE DUMMY ELEMENT
    X = np.delete(X, 0, 0)
    label = np.delete(label, 0)
    #CLUSTERING
    clustering = AgglomerativeClustering(linkage='single', n_clusters=3).fit(X)
    print(f'Label is:{clustering.labels_}')
    #print(f'Array is:{X}')
    #print(f'Label is:{label}')