import numpy
import csv

class dbscan:
    def __init__(self, training_data):
        self.training_data = training_data
    
    def fit(self, eps, min_p):
        self.eps = eps
        self.min_p = min_p

        labels = [0]*len(self.training_data)

        C = 0
        for P in range(0, len(self.training_data)):
            if not (labels[P] == 0):
                continue
            
            NeighborPts = self.getNeighbors(P, eps)
            if len(NeighborPts) < min_p:
                labels[P] = -1
            else: 
                C += 1
                self.expand(labels, P, NeighborPts, C, eps, min_p)

        self.labels = numpy.array(labels)

    def predict(self, test_data, test_data_label):
        # temukan label baru berdasarkan label training
        labels = []

        for P in range(0, len(test_data)):
            neighbors_labels = []

            for Pn in range(0, len(self.training_data)):
                if numpy.linalg.norm(self.training_data[Pn] - test_data[P]) < self.eps:
                    neighbors_labels.append(self.labels[Pn])
            
            if len(neighbors_labels) == 0:
                labels.append(-1)
            elif len(neighbors_labels) == 1:
                labels.append(neighbors_labels[0])
            else:
                labels.append(max(set(neighbors_labels), key = neighbors_labels.count) )

        print('Real Data Test Label:\n', test_data_label)
        print('Predicted test Label:\n', labels)

        # tentukan akurasi, dibandingkan dengan test_data_label
        correct_predict = 0
        for Q in range(0, len(test_data_label)):
            if (labels[Q] == -1):
                pass
            elif (test_data_label[Q] + 1 == labels[Q]):
                correct_predict += 1

        print('Accuracy: ', correct_predict / len(test_data_label) * 100, ' percent')

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
        
        for Pn in range(0, len(self.training_data)):
            if numpy.linalg.norm(self.training_data[P] - self.training_data[Pn]) < eps:
                neighbors.append(Pn)
                
        return neighbors
