# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:40:26 2013

@author: Shane Grigsby <shane@geog.ucsb.edu>
"""

##OPTICKS Final

import sys
import scipy
import pickle
from scipy.spatial import cKDTree
from numba import autojit
from itertools import izip

from sklearn.neighbors import BallTree


class setOfObjects(BallTree):    

    def __init__(self,data_points):     

        super(setOfObjects,self).__init__(data_points)

        self._n             =   len(self.data)
        self._processed     =   scipy.zeros((self.n,1),dtype=bool)
        self._reachability  =   scipy.ones(self.n)*scipy.inf
        self._core_dist     =   scipy.ones(self.n)*scipy.nan
        self._index         =   scipy.array(range(self.n))
        self._nneighbors    =   scipy.ones(self.n,dtype=int)

    def setneighborhood(self,point,epsilon):
        neighborhood     =  self.query(self.data[point],self.n,
                                        distance_upper_bound=epsilon)
        self._nneighbors[point] = scipy.isfinite(neighborhood[0]).nonzero()[0].size
        
    def set_core_dist(self,point,MinPts):
        core_dist       = self.query(point,MinPts)

    def __iter__(self):
        output = (self._index[n] for n in self._index if not self._processed[self._index[n]])

@autojit
def prep_optics(SetofObjects,epsilon,MinPts):
    for i in xrange(SetofObjects._n):
        SetofObjects.setneighborhood(i,epsilon)
        SetofObjects.setcoredist(i,MinPts)
    print(str(SetofObjects.n) + ' points processed')

def build_optics(SetOfObjects,epsilon,MinPts,Output_file_name):
    for point in xrange(SetOfObjects.n):
        if SetOfObjects._processed[point] == False:
            expandClusterOrder(SetOfObjects,point,epsilon,
                               MinPts,Output_file_name)
                               
def expandClusterOrder(SetOfObjects,point,epsilon,MinPts,Output_file_name):
    core_dist = SetOfObjects.query(SetOfObjects.data[point],MinPts,
                                   distance_upper_bound=epsilon)[0][MinPts-1]
    SetOfObjects._reachability[point]= core_dist
    nobjects = 0
    if scipy.isfinite(SetOfObjects._reachability[point]):
        while not SetOfObjects._processed[point]:
            SetOfObjects._processed[point] = True
            with open(Output_file_name, 'a') as file:
                file.write((str(point) + ', ' + str(SetOfObjects._reachability[point]) + '\n'))
                point = set_reach_dist(SetOfObjects,point,epsilon)
        print('Object #' + str(nobjects) + 'Found!')
        nobjects = nobjects + 1

def set_reach_dist(SetOfObjects,point_index,epsilon):
    distances, indices = SetOfObjects.query(SetOfObjects.data[point_index],
                                            SetOfObjects._nneighbors[point_index],
                                            distance_upper_bound=epsilon)
    if scipy.iterable(distances):
        if scipy.isfinite(distances[-1]):
            c_dist = distances[-1]
            unprocessed = SetOfObjects._index[(scipy.where(test_set._processed < 1)[0])]
            SetOfObjects._reachability[unprocessed] = scipy.minimum(SetOfObjects._reachability[unprocessed],c_dist)
            return unprocessed[0]
        else: 
            return point_index
    else:
        return point_index
    
###setting up testing data
testdata = scipy.rand(5000,3)
test_set = setOfObjects(testdata)
#neighbors = test_set.query(test_set.data[1337],test_set.n,distance_upper_bound=.07)
#num_neighbors = len(scipy.isfinite(neighbors[0]).nonzero()[0])
#neighbors,distances=neighbors[1][:num_neighbors],neighbors[0][:num_neighbors]
#core_dist = test_set.query(test_set.data[1337],num_neighbors,distance_upper_bound=.07)[0][num_neighbors-1]
#centerPoint = 1337
def test_numba(kdtree,epsilon):
    neighbors = scipy.ones(kdtree.n)
    for i in range(kdtree.n):
        debugg = test_set.query(kdtree.data[i],kdtree.n,distance_upper_bound=epsilon)
        neighbors[i]=scipy.isfinite(debugg[0]).nonzero()[0].size
    return neighbors