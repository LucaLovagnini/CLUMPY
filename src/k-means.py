import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class CLUMPY:

    def __init__(self, k, file):
        # input file
        self.__file = file
        self.__k = k
        # we generate k different colors, one per centroid
        self.__colors = cm.rainbow(np.linspace(0, 1, self.__k))
        print("k=", k)
        print("Reading {}...".format(file))
        # data points
        self.__points = np.loadtxt(file)
        # number of data points
        self.__n = self.__points.shape[0]
        # data points dimensionality (or length according to numpy terminology)
        if self.__n < k:
            raise ValueError("Number of clusters k={} is smaller than number of data points n={}".format(k, self.__n))
        self.__d = self.__points.shape[1]
        if self.__d < 2:
            raise ValueError("data points must have at least two dimensions")
        print("Read {}: {} points in {} dimensions.".format(file, self.__n, self.__d))
        # __class[i] = j : the i-th data point is assigned to the j-th cluster
        self.__class = np.zeros(self.__n, dtype=np.int8)
        print("Generating seeds...")
        # __centroids[i] = centroid of length d
        self.__centroids = np.empty((k, self.__d))
        # generate k random indexes
        random_indexes = list(range(self.__n))
        random.shuffle(random_indexes)
        # we decide centroids by randomly picking up data points
        for i in range(k):
            self.__centroids[i] = self.__points[random_indexes[i]]
        # __distances[i] = d : the i-th data point is distant d from its centroid (i.e., __class[i])
        self.__distances = np.full(self.__n, math.inf)
        self.assign_datapoints()

    def assign_datapoints(self):
        # for each datapoint
        for i, point in enumerate(self.__points):
            min_distance_index = float('nan')
            min_distance = math.inf
            # for each centroid
            for j, centroid in enumerate(self.__centroids):
                # compute the euclidean distance between the i-th point and the j-th centroid
                d = np.linalg.norm(point - centroid)
                if d < min_distance:
                    min_distance_index = j
                    min_distance = d
            # update cluster assignment and distances
            if not math.isnan(min_distance_index):
                self.__class[i] = min_distance_index
                self.__distances[i] = min_distance

    def plot(self):
        for i, [X, Y] in enumerate(self.__points):
            color = self.__colors[self.__class[i]]
            if [X, Y] in self.__centroids.tolist():
                plt.scatter(X, Y, c=color, marker='^')
            else:
                plt.scatter(X, Y, c=color)
        plt.show()


if __name__ == "__main__":
    clumpy = CLUMPY(5, "datasets/202d")
    clumpy.plot()
