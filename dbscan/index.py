import numpy
import csv

class Dbscan:
    def __init__(self, filename):
        data = numpy.zeros((1,4))
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    data = numpy.append(data, numpy.array([[float(row[0]),float(row[1]),float(row[2]),float(row[3])]]), axis=0)
                    line_count += 1
            data = numpy.delete(data, 0, 0)
            self.data = data
    
    def dbscan(self, eps, MinPts):
        labels = [0]*len(self.data)

        C = 0
        for P in range(0, len(self.data)):
            if not (labels[P] == 0):
                continue
            
            NeighborPts = self.regionQuery(P, eps)
            if len(NeighborPts) < MinPts:
                labels[P] = -1
            else: 
                C += 1
                self.growCluster(labels, P, NeighborPts, C, eps, MinPts)
            
        return labels


    def growCluster(self, labels, P, NeighborPts, C, eps, MinPts):
        labels[P] = C
        
        i = 0
        while i < len(NeighborPts):    
            Pn = NeighborPts[i]
            if labels[Pn] == -1:
                labels[Pn] = C

            elif labels[Pn] == 0:
                labels[Pn] = C
                PnNeighborPts = self.regionQuery(Pn, eps)
                
                if len(PnNeighborPts) >= MinPts:
                    NeighborPts = NeighborPts + PnNeighborPts
            i += 1        

    def regionQuery(self, P, eps):
        neighbors = []
        
        for Pn in range(0, len(self.data)):
            if numpy.linalg.norm(self.data[P] - self.data[Pn]) < eps:
                neighbors.append(Pn)
                
        return neighbors


dbscan = Dbscan('iris.csv')

labels = dbscan.dbscan(0.5, 5)
print(labels)