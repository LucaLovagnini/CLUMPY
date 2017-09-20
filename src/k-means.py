import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import multiprocessing as mp

from sklearn.decomposition import PCA as sklearnPCA


def generate_clusters (k, n, d, max_value=30000, deviation=10000):
    # generate n random points in d dimensions with elements in [-deviation, deviation]
    points = np.random.np.random.uniform(low=-deviation, high=deviation, size=(n, d))
    # generate k points in d dimensions with elements in [0, max_value]
    centers = np.random.random((k, d))*max_value
    # generate clusters: for each point, randomly select a center and add to it
    print(points, centers)
    for i, point in enumerate(points):
        points[i] += random.choice(centers)
    return points


centroids = np.full(1, -1)
points = np.full(1, -1)


def assign_point(i):
    min_distance_index = float('nan')
    min_distance = math.inf
    # for each centroid
    for j, centroid in enumerate(centroids):
        # compute the euclidean distance between the i-th point and the j-th centroid
        d = np.linalg.norm(points[i] - centroid)
        if d < min_distance:
            min_distance_index = j
            min_distance = d
    # update cluster assignment and distances
    return min_distance, min_distance_index


class CLUMPY:

    def __init__(self, k, file=None, n=None, d=None, delimiter=None, iterations=100):
        global centroids, points
        self.__file = file                      # input file
        self.__k = k                            # number of clusters
        self.__iterations = iterations          # number of iterations
        self.__colors = \
            cm.rainbow(np.linspace(0, 1, self.__k))     # colors[i] = color of the i-th cluster
        if file:
            # if file is specified, read points from file
            print("Reading {}...".format(file))
            points = np.loadtxt(file, delimiter=delimiter)        # data points
        else:
            # otherwise generate n clusterized points in d dimensions
            if not n or not d:
                raise ValueError("missing n={} or d={}".format(n, d))
            self.__n = n
            self.__d = d
            print("Generating {} random points in {} dimensions...".format(n, d))
            points = generate_clusters(k, n, d)
        self.__d = points.shape[1]       # points dimensions
        self.__n = points.shape[0]       # number of data points
        centroids = np.empty((k, self.__d))             # centroids[i] = i-th centroid vector
        print(centroids)
        # class[i] = j : the i-th data point is assigned to the j-th cluster
        self.__class = np.full(self.__n, -1, dtype=np.int16)
        # energy[i] : energy of the i-th cluster
        self.__energy = np.zeros(k)
        self.__distances = np.zeros(self.__n)
        self.__clusters_size = np.zeros(self.__k, dtype=np.int32) # number of points assigned to each cluster
        self.__clusters_sum = np.zeros((k, self.__d)) # sum of all vectors assigned to each cluster
        # sanity checks
        if self.__n < k:
            raise ValueError("Number of clusters k={} is smaller than number of data points n={}".format(k, self.__n))
        if self.__d < 2:
            raise ValueError("data points must have at least two dimensions")
        print("{} points in {} dimensions.".format(self.__n, self.__d))
        print("Generating seeds...")
        # generate k random indexes
        random_indexes = list(range(self.__n))
        random.shuffle(random_indexes)
        # we decide centroids by randomly picking up data points
        for i in range(k):
            centroids[i] = points[random_indexes[i]]
        #self.plot()

    def plot(self):
        print("Plotting...")
        points_to_plot = np.concatenate((centroids, points), axis=0)
        if self.__d > 2:
            point_norm = (points_to_plot - points_to_plot.min())/(points_to_plot.max() - points_to_plot.min())
            pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
            points_to_plot = np.array(pca.fit_transform(point_norm))
        for i, (X, Y) in enumerate(points_to_plot):
            if i<self.__k:
                plt.scatter(X, Y, c=self.__colors[i], s=100, marker="^")
            else:
                plt.scatter(X, Y, c=self.__colors[self.__class[i-self.__k]])
        plt.show()

    def cluster(self):
        global centroids
        for iteration in range(self.__iterations):
            # update each assignment: if no point changed its cluster, then we have reached the optimum
            # for each datapoint
            t = time.time()
            #results = [assign_point(i) for i in range(self.__n)]
            pool = mp.Pool(mp.cpu_count()-1)
            results = pool.map(assign_point, range(self.__n))
            assignment_time = time.time() - t
            t = time.time()
            for i, (min_distance, min_distance_index) in enumerate(results):
                # update cluster assignment and distances
                if min_distance_index != self.__class[i]:
                    if self.__class[i] != -1:
                        self.__clusters_size[self.__class[i]] -= 1
                        self.__clusters_sum[self.__class[i]] -= points[i]
                        self.__energy[self.__class[i]] -= self.__distances[i]
                    self.__class[i] = min_distance_index
                    self.__clusters_size[min_distance_index] += 1
                    self.__clusters_sum[min_distance_index] += points[i]
                    self.__distances[i] = min_distance ** 2
                    self.__energy[min_distance_index] += self.__distances[i]
            # update centroids
            centroids_unchanged = True
            for i in range(self.__k):
                new_centroid = self.__clusters_sum[i] / self.__clusters_size[i]
                centroids_unchanged = centroids_unchanged and np.array_equal(new_centroid, centroids[i])
                centroids[i] = new_centroid
            print("iteration {} assignment={} updating={}".format(iteration, assignment_time, time.time() - t))
            if centroids_unchanged:
                print("Centroids unchanged, terminating...")
                break
        else:
            print("All iterations are finished")
        self.plot()


if __name__ == "__main__":
    clumpy = CLUMPY(k=5, d=64, n=10000)
    clumpy.cluster()
