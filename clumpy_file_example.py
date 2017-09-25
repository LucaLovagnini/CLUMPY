from clumpy import CLUMPY

# create a default CLUMPY object (i.e., 2000 points in 2D) and 5 clusters
clumpy = CLUMPY(5, file="datasets/202d")
#create clusters
clumpy.cluster()
#show clusters
clumpy.plot()