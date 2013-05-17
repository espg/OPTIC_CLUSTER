# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 23:12:16 2013

@author: grigsbye
"""

####Python Plotting Code####

# Plot result
import pylab as pl
from itertools import cycle
import mpl_toolkits.mplot3d.axes3d as p3

pl.close('all')
ax = p3.Axes3D(pl.figure(1))

# Black removed and is used for noise instead.
colors = cycle('bgrcmybgrcmybgrcmybgrcmy')
for k, col in zip(set(labels), colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        markersize = 6
    class_members = [index[0] for index in numpy.argwhere(labels == k)]
    cluster_core_samples = [index for index in core_samples
                            if labels[index] == k]
    for index in class_members:
        xt = data[index]      ### I changed this from 'd' to 'data'...
        if index in core_samples and k != -1:
            markersize = 14
        else:
            markersize = 6
        ax.plot3D(ravel(xt[0]), ravel(xt[1]),ravel(xt[2]), 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=markersize)

pl.title('Estimated number of clusters: %d' % n_clusters_)


from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
import tables
import scipy
from scipy import stats
import palm
from scipy import spatial
from scipy import linalg
from numpy import array
from numpy import newaxis
import numpy
from math import acos
import math

input = './good_12_leafangle.h5'
h5f = tables.openFile(input,'r')
lidar = h5f.root.data

# (x < 52035000) sql for half the plot

subset = lidar.readWhere('(Classification ==1)&(scanNumber == 1)&(Range < 5)')
Set = palm.list_arrays(subset)
id_set = Set[0]
	

X = lidar.cols.x[:]
X = X[id_set]
X = numpy.float32(X)
X = X/10000
X = X + 290000
Y = lidar.cols.y[:]
Y = Y[id_set]
Y = numpy.float32(Y)
Y = Y/10000
Y = Y + 3900000
Z = lidar.cols.z[:]
Z = Z[id_set]
Z = numpy.float32(Z)
Z = Z/10000

data = [X,Y,Z]
data = numpy.transpose(data)

#############################################################################
#Split data into tiles to pass through DBSCAN

sub = 100

d_split = numpy.array_split(data,sub)

# Density based scan using sklearn toolbox: Example taken from "http://scikit-learn.org/
#stable/auto_examples/cluster/plot_dbscan.html#example-cluster-plot-dbscan-py"

from sklearn.cluster import DBSCAN
from scipy.spatial import distance

##############################################################################
# Compute similarities
d = d_split[1]
D = distance.squareform(distance.pdist(d))
S = 1 - (D / numpy.max(D))

##############################################################################
# Compute DBSCAN
db = DBSCAN().fit(S, eps= 0.35, min_samples=5)
core_samples = db.core_sample_indices_
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print 'Estimated number of clusters: %d' % n_clusters_


##############################################################################
# Plot result
import pylab as pl
from itertools import cycle
import mpl_toolkits.mplot3d.axes3d as p3

pl.close('all')
ax = p3.Axes3D(pl.figure(1))
#pl.figure(1)
#pl.clf()

# Black removed and is used for noise instead.
colors = cycle('bgrcmybgrcmybgrcmybgrcmy')
for k, col in zip(set(labels), colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        markersize = 6
    class_members = [index[0] for index in numpy.argwhere(labels == k)]
    cluster_core_samples = [index for index in core_samples
                            if labels[index] == k]
    for index in class_members:
        xt = data[index]
        if index in core_samples and k != -1:
            markersize = 14
        else:
            markersize = 6
        ax.plot3D(ravel(xt[0]), ravel(xt[1]),ravel(xt[2]), 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=markersize)

pl.title('Estimated number of clusters: %d' % n_clusters_)
#pl.show()
