from dbscan_implementation import DbscanImplementation

dbscan_implementation = DbscanImplementation('iris.csv')

labels = dbscan_implementation.dbscan(0.5, 5)
print(labels)