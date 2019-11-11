import numpy
import csv

class dbscan:
    def __init__(self, filename):
        data = numpy.zeros((1,4))
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    data = numpy.append(data, numpy.array([[float(row[0]),float(row[1]),float(row[2]),float(row[3])]]), axis=0)
                    line_count += 1
            data = numpy.delete(data, 0, 0)
            self.data = data
    
    def fit(self, eps, min_p):
        labels = [0]*len(self.data)

        C = 0
        for P in range(0, len(self.data)):
            if not (labels[P] == 0):
                continue
            
            NeighborPts = self.getNeighbors(P, eps)
            if len(NeighborPts) < min_p:
                labels[P] = -1
            else: 
                C += 1
                self.expand(labels, P, NeighborPts, C, eps, min_p)
            
        return numpy.array(labels)


    def expand(self, labels, P, NeighborPts, C, eps, min_p):
        labels[P] = C
        
        i = 0
        while i < len(NeighborPts):    
            Pn = NeighborPts[i]
            if labels[Pn] == -1:
                labels[Pn] = C

            elif labels[Pn] == 0:
                labels[Pn] = C
                PnNeighborPts = self.getNeighbors(Pn, eps)
                
                if len(PnNeighborPts) >= min_p:
                    NeighborPts = NeighborPts + PnNeighborPts
            i += 1        

    def getNeighbors(self, P, eps):
        neighbors = []
        
        for Pn in range(0, len(self.data)):
            if numpy.linalg.norm(self.data[P] - self.data[Pn]) < eps:
                neighbors.append(Pn)
                
        return neighbors
