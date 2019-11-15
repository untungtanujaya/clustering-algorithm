import math

class KMeans:
    def __init__(self, data, k):
        self.vectors =  self.make_vectors(data)
        self.k = k
        self.centroids = self.make_centroids()
        self.clusters = self.make_clusters()
        self.groups = self.make_groups()

    def make_vectors(self, data):
        vectors = []
        for i, row in enumerate(data):
            if i>0:
                vector = dict()
                vector["centroid"] = 0
                vector["key"] = i
                vector["value"] = row[0], row[1], row[2], row[3]
                vector["group"] = row[4]
                vectors.append(vector)
        return vectors

    def make_centroids(self):
        c = [0 for i in range(len(self.vectors[0]['value']))]
        for vector in self.vectors:
            for j, el in enumerate(c):
                c[j] += float(vector["value"][j])
        for i in range(len(self.vectors[0]['value'])):
            c[i] = c[i]/len(self.vectors)
        centroids = []
        for j in range(int(self.k)):
            max_d = 0
            max_i = 0
            for i, vector in enumerate(self.vectors):
                if len(centroids) > 0:
                    if not any(el.get("value") == vector["value"] for el in centroids):
                        dist = self.count_distance(vector["value"], centroids[j-1]["value"])
                else:
                    dist = self.count_distance(vector["value"], c)    
                if max_d < dist:
                    max_d = dist
                    max_i = i
            centroid = dict()
            centroid["centroid"] = j
            centroid["value"] = self.vectors[max_i]["value"]
            centroids.append(centroid)
        return centroids

    def count_distance(self, a, b):
        if len(a) == len(b):
            result = 0
            for i in range(len(a)):
                dist = float(a[i]) - float(b[i])
                dist = dist * dist
                result += dist
            return math.sqrt(result)    
        else:
            return -9999    

    def make_clusters(self):
        result = []
        is_change = True
        while is_change:
            is_change = False
            for i, vector in enumerate(self.vectors):
                closest_centroid = self.find_closest_centroid(vector)
                if closest_centroid != vector["centroid"]:
                    is_change = True
                    self.vectors[i]["centroid"] = closest_centroid
            if is_change:
                self.update_centroids()
        for centroid in self.centroids:
            cluster_points = ([(x['key'], x['value'], x['group']) for x in self.vectors if x['centroid'] == centroid['centroid']])
            result.append(cluster_points)
        return result    

    def find_closest_centroid(self, point):
        min_d = float("inf")
        min_centroid = -1
        for i, centroid in enumerate(self.centroids):
            dist = self.count_distance(centroid["value"], point["value"])
            if min_d > dist:
                min_d = dist
                min_centroid = centroid["centroid"]
        return min_centroid

    def make_groups(self):
        result = {}
        for i, centroid in enumerate(self.centroids):
            groups = ([int(x['group']) for x in self.vectors if x['centroid'] == centroid['centroid']])
            avg = sum(groups)/len(groups)
            result[i] = round(avg)
        return result

    def update_centroids(self):
        c = [[0 for i in range(len(self.vectors[0]['value']))] for j in range(self.k)]
        for vector in self.vectors:
            centroid_j = vector["centroid"]
            for j, el in enumerate(c[centroid_j]):
                c[centroid_j][j] += float(vector['value'][j])
        for centroid_j, el in enumerate(c):
            centroid_count = len([d for d in self.vectors if d['centroid']==centroid_j])
            if centroid_count != 0:
                for i in range(len(self.vectors[0]['value'])):
                    c[centroid_j][i] = c[centroid_j][i]/centroid_count
        for i, centroid in enumerate(self.centroids):
            self.centroids[i]['value'] = c[i]

    def predict(self, test, labeltest):
        result = []
        print(f'Real Data Test Label: {labeltest}')
        for point in test:
            min_d = float('inf')
            min_i = -1
            for i, centroid in enumerate(self.centroids):
                dist = self.count_distance(centroid["value"], point)
                if min_d > dist:
                    min_d = dist
                    min_i = i
            result.append(self.groups[min_i])
        print(f'Predicted Test Label: {result}')
        count = 0
        for i in range(len(labeltest)):
            if labeltest[i] == result[i]:
                count += 1
        return count*100/len(labeltest)            