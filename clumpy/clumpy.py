import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing as mp
import SharedArray as sa
import random
import math
import time

from sklearn.decomposition import PCA as sklearnPCA


def generate_clusters (k, n, d, max_value, deviation):
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

    def __init__(
            self,
            k,
            file=None,
            n=2000,
            d=2,
            max_value=600000,
            deviation=400000,
            delimiter=None,
            iterations=100,
            random_centroids=False,
            processes = mp.cpu_count()):
        """
        CLUMPY constructor.
        :param k: number of clusters
        :param file: input file. When None, generates random clusters. (default: None)
        :param n: used when file=None. Number of generated points. (default: 2000)
        :param d: used when file=None. Generated point's dimensions. (default: 2)
        :param max_value: used when file=None. Max value for generated clusters centers (default: 600000)
        :param deviation: used when file=None. Max distance from cluster center for each point (default: 400000)
        :param delimiter: used when file!=None, used for numpy.loadtxt() (default: None)
        :param iterations: number of k-means iterations (default: 100)
        :param random_centroids: if True generate random seeds.If False, use k-means++ (serial version) (default: False)
        :param processes: number of spawned processes (default: multiprocessing.cpu.count())

        """
        global centroids, points
        try:
            sa.delete("centroids")
        except FileNotFoundError:
            pass
        try:
            sa.delete("points")
        except FileNotFoundError:
            pass

        if max_value is None:
            max_value = 600000
        if deviation is None:
            deviation = 400000
        if iterations is None:
            iterations = 100
        if random_centroids is None:
            random_centroids = False
        if processes is None:
            mp.cpu_count()
            
        self.__file=file
        self.__k = k
        self.__iterations = iterations
        self.__random_centroids = random_centroids
        self.__colors = \
            cm.rainbow(np.linspace(0, 1, self.__k))     # colors[i] = color of the i-th cluster
        self.__processes = processes
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
            points_copy = generate_clusters(k, self.__n, d, max_value, deviation)
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
        with mp.Pool(self.__processes) as pool:
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
                print("Iterations finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("k", help="number of clusters", type=int)
    parser.add_argument("--file", help="input file. If not set, generates random clusters")
    parser.add_argument("--n", help="used when file is not set. Number of generated points. (default: 2000)", type=int)
    parser.add_argument("--d", help="used when file is not set. Generated point's dimensions. (default: 2)", type=int)
    parser.add_argument("--max_value",
                        help="used when file is not set. Max value for generated clusters centers (default: 600000)",
                        type=int)
    parser.add_argument("--deviation",
                        help="used when file is not set. Max distance "
                             "from cluster center for each point(default: 400000)",
                        type=int)
    parser.add_argument("--delimiter", help="used when file is set, used for numpy.loadtxt()")
    parser.add_argument("--iterations", help="number of k-means iterations (default: 100)", type=int)
    parser.add_argument("--random_centroids", help="if True generate random seeds."
                        "If False, use k-means++ (serial version) (default: False)", type=bool)
    parser.add_argument("--processes", help="number of spawned processes (default: multiprocessing.cpu.count())",
                        type=int)
    args = parser.parse_args()

    k = args.k
    random_centroids = args.random_centroids
    processes = args.processes

    if args.file:
        clumpy = CLUMPY(k=k, file=args.file, delimiter=args.delimiter,
                        random_centroids=random_centroids, processes=processes)
    else:
        clumpy = CLUMPY(k=k, n=args.n, d=args.d, max_value=args.max_value, deviation=args.deviation,
                        delimiter=args.delimiter, iterations=args.iterations, random_centroids=random_centroids,
                        processes=processes)

    clumpy.cluster()
    clumpy.plot()
