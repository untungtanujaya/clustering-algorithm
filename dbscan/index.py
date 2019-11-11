from sklearn.cluster import DBSCAN
import numpy
import csv

from dbscan_implementation import dbscan

data = numpy.zeros((1,4))
with open('iris.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    line_count = 0
    
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            data = numpy.append(data, numpy.array([[float(row[0]),float(row[1]),float(row[2]),float(row[3])]]), axis=0)
            line_count += 1
    data = numpy.delete(data, 0, 0)

eps = 0.5
min_p = 5

# Implementation

dbscan = dbscan(data)
labels = dbscan.fit(eps, min_p)
print('from implementation\n', numpy.array(labels))

# Scikit

dbscan_scikit = DBSCAN(eps=eps, min_samples=min_p).fit(data)
print('from scikit\n', dbscan_scikit.labels_)