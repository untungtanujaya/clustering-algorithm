#MAIN FUNGSI
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

#Pilihan K-means, Agglomerative, DBScan

    #K-means
        #Train and Test

    #Agglomerative
    #Pilih Linkage
        #Single
            #Train and Test
        #Complete
            #Train and Test
        #Average
            #Train and Test
        #Average-Group
            #Train and Test
    
    #DBScan
        #Train and Test