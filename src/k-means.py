import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import multiprocessing as mp
import SharedArray as sa

from sklearn.decomposition import PCA as sklearnPCA


def generate_clusters (k, n, d, max_value=600000, deviation=80000):
    # generate n random points in d dimensions with elements in [-deviation, deviation]
    points = np.random.uniform(low=-deviation, high=deviation, size=(n, d))
    # generate k points in d dimensions with elements in [0, max_value]
    centers = np.random.random((k, d))*max_value
    # generate clusters: for each point, randomly select a center and add to it
    for i, point in enumerate(points):
        points[i] += random.choice(centers)
    return points


try:
    centroids = sa.attach("shm://centroids")
except FileNotFoundError:
    centroids = sa.create("shm://centroids", 1)
try:
    points = sa.attach("shm://points")
except FileNotFoundError:
    points = sa.create("shm://points", 1)


def assign_point(i, k = len(centroids)):
    min_distance_index = float('nan')
    min_distance = math.inf
    # for each centroid
    for j, centroid in enumerate(centroids):
        if j > k:
            break
        # compute the euclidean distance between the i-th point and the j-th centroid
        d = np.linalg.norm(points[i] - centroid)
        if d < min_distance:
            min_distance_index = j
            min_distance = d
    # update cluster assignment and distances
    return min_distance, min_distance_index


class CLUMPY:

    def __init__(self, k, file=None, n=None, d=None, delimiter=None, iterations=100, random_centroids=False):
        global centroids, points
        try:
            sa.delete("centroids")
        except FileNotFoundError:
            pass
        try:
            sa.delete("points")
        except FileNotFoundError:
            pass
        self.__file = file                          # input file
        self.__k = k                                # number of clusters
        self.__iterations = iterations              # number of iterations
        self.__random_centroids = random_centroids  # if false, use k-means++
        self.__colors = \
            cm.rainbow(np.linspace(0, 1, self.__k))     # colors[i] = color of the i-th cluster

        if file:
            # if file is specified, read points from file
            print("Reading {}...".format(file))
            points_copy = np.loadtxt(file, delimiter=delimiter)        # data points
            (self.__n, self.__d) = points_copy.shape
        else:
            # otherwise generate n clusterized points in d dimensions
            if not n or not d:
                raise ValueError("missing n={} or d={}".format(n, d))
            self.__n = n
            self.__d = d
            print("Generating {} random points in {} dimensions...".format(n, d))
            points_copy = generate_clusters(k, self.__n, d)
        print(self.__n, self.__d)
        centroids = sa.create("shm://centroids", (k, self.__d))
        points = sa.create("shm://points", (self.__n,self.__d))
        np.copyto(points, points_copy)
        self.__d = points.shape[1]       # points dimensions
        self.__n = points.shape[0]       # number of data points
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
        if random_centroids:
            # generate k random indexes
            random_indexes = list(range(self.__n))
            random.shuffle(random_indexes)
            # we decide centroids by randomly picking up data points
            for i in range(k):
                centroids[i] = points[random_indexes[i]]
        else:
            # k-means++
            # step 1: select a random point as first centroid
            centroids[0] = random.choice(points)
            for i in range(1,k):
                # step 2: asssign each point to the closest centroid (read only the first i elements in centroids)
                results = [assign_point(j, i) for j in range(self.__n)]
                # step 3: normalize distances
                distances = [result[0] for result in results]
                norm = [d/sum(distances) for d in distances]
                # step 4: generate an uniform random number in [0,1]
                r = np.random.uniform()
                # step 5: choose the first element in norm that is greater or equal than r
                acc = 0
                chosen_index = 0
                for n in norm:
                    acc += n
                    if acc >= r:
                        break
                    chosen_index += 1
                centroids[i] = points[chosen_index]

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
        with mp.Pool(1) as pool:
            for iteration in range(self.__iterations):
                # update each assignment: if no point changed its cluster, then we have reached the optimum
                # for each datapoint
                t = time.time()
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
    clumpy = CLUMPY(k=5, n=5000, d=5)
    clumpy.cluster()
