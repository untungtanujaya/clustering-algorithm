from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
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
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=3).fit(X)
    print(f'Label is:{clustering.labels_}')
    #print(f'Label is:{label}')
    #ENCODE Label to numeric
    le = LabelEncoder()
    le.fit(label)
    label_num = le.transform(label)
    print(f'Label_num is:{label_num}')
    #SCORING
    print("%.6f" % v_measure_score(label_num, clustering.labels_))
    print("%.6f" % completeness_score(label_num, clustering.labels_))
    print("%.6f" % homogeneity_score(label_num, clustering.labels_))
    #print(f'Array is:{X}')
    #print(f'Label is:{label}')