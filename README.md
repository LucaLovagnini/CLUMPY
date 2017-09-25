# CLUMPY: CLUstering with Multicores and PYthon
CLUMPY is a parallel implementation of the Lloyd's k-means algorithm.

Clustering with CLUMPY is incredibly easy, as shown in [this](https://github.com/LucaLovagnini/CLUMPY/blob/master/clumpy_example.py) example:

	from clumpy import CLUMPY

	# create a default CLUMPY object (i.e., 2000 points in 2D) and 5 clusters
	clumpy = CLUMPY(5)
	#create clusters
	clumpy.cluster()
	#show clusters
	clumpy.plot()
	
The example above will generate 2000 random points in 2 dimensions and create 5 clusters from them, which will be then plotted.

Do you want to use your own file containg points in any dimension? No problem, as shown in [this](https://github.com/LucaLovagnini/CLUMPY/blob/master/clumpy_file_example.py) example:

	from clumpy import CLUMPY

	# create a default CLUMPY object (i.e., 2000 points in 2D) and 5 clusters
	clumpy = CLUMPY(5, file="datasets/202d")
	#create clusters
	clumpy.cluster()
	#show clusters
	clumpy.plot()

You can see the resulting clusters from the first example below:

![](https://github.com/LucaLovagnini/CLUMPY/blob/master/figures/example.png)

## Requirements:

In order to make CLUMPY works you will need:

  - Linux OS
  - Python 3
  - Numpy
  - Matplotlib
  - [SharedArray](https://github.com/ddboline/shared-array)
  - A multi-core architecture (optional, but highly reccomended)

## User Manual:

Seed initialization methods implemented in CLUMPY are:

  1. Random points.
  2. [K-means++](ilpubs.stanford.edu/778/1/2006-13.pdf)

Termination conditions are:

  1. Unchanged centroids.
  2. Number of iterations.
  3. Clusters energy below a given threshold (todo).

Files are read using [`numpy.loadtxt()`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html), where you can specify your own `delimiter` (see below for details).

