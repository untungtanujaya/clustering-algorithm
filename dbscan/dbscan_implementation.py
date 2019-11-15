import numpy

class dbscan:
    # Melakukan inisiasi training data pada class
    def __init__(self, training_data):
        self.training_data = training_data
    
    # Melakukan training dengan menerima masukan
    # eps: jarak minimal antara 2 sample agar dikatakan bertetangga
    # min_p: jumlah tetangga minimal yang harus dimiliki suatu sample agar dikatakan core point
    def fit(self, eps, min_p):
        self.eps = eps
        self.min_p = min_p

        labels = [0]*len(self.training_data)

        cluster = 0
        for training_data_index in range(0, len(self.training_data)):
            if not (labels[training_data_index] == 0):
                continue
            
            neighbors_points = self.getNeighbors(training_data_index)
            if len(neighbors_points) < min_p:
                labels[training_data_index] = -1
            else: 
                cluster += 1
                self.expand(labels, training_data_index, neighbors_points, cluster)

        self.labels = numpy.array(labels)

    # Melakukan test berdasarkan hasil training sebelumnya 
    def predict(self, test_data, test_data_label):
        # temukan label baru berdasarkan label training
        labels = []

        for test_data_index in range(0, len(test_data)):
            neighbors_labels = []

            for training_data_index in range(0, len(self.training_data)):
                if numpy.linalg.norm(self.training_data[training_data_index] - test_data[test_data_index]) < self.eps:
                    neighbors_labels.append(self.labels[training_data_index])
            
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
        for test_data_label_index in range(0, len(test_data_label)):
            if (labels[test_data_label_index] == -1):
                pass
            elif (test_data_label[test_data_label_index] + 1 == labels[test_data_label_index]):
                correct_predict += 1

        print('Accuracy: ', correct_predict / len(test_data_label) * 100, ' percent')

    # Melakukan pencarian keseluruhan anggota yang sama (satu cluster) 
    def expand(self, labels, training_data_index, neighbors_points, cluster):
        labels[training_data_index] = cluster
        
        i = 0
        while i < len(neighbors_points):    
            point_index = neighbors_points[i]
            if labels[point_index] == -1:
                labels[point_index] = cluster

            elif labels[point_index] == 0:
                labels[point_index] = cluster
                current_neighbors_points = self.getNeighbors(point_index)
                
                if len(current_neighbors_points) >= self.min_p:
                    neighbors_points += current_neighbors_points
            i += 1        

    # Mendapatkan tetangga untuk sample pada suatu index
    def getNeighbors(self, index):
        neighbors = []
        
        for training_data_index in range(0, len(self.training_data)):
            if numpy.linalg.norm(self.training_data[index] - self.training_data[training_data_index]) < self.eps:
                neighbors.append(training_data_index)
                
        return neighbors
