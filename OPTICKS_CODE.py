# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 17:10:49 2012

@author: -
"""
import sys
import scipy
import pickle
from scipy.spatial import cKDTree
from numba import autojit
from itertools import izip


#outnam =       ### what is ths supposed to be?

### Superclassing the CKDTree
## Explained in too much detail
class setOfObjects(cKDTree):    
# Defines the name of the superclass, and that class that is being inherited from
    def __init__(self,data_points):     
# Initaizes the object, provides 'self' for instances of the class AND,
# defines input to the class-- which is DATA_POINTS
# Earlier versions of this code had 'processed' listed, leading to much confusion

        super(setOfObjects,self).__init__(data_points)
# Defines which object methods to bind to which method. The stuff inside of 
# the super() parentheis is totally redundant and confusing-- it saids bind 
# the parent class to setOfObjects and self-- of course that's what we want.
# Part of the reason that this was so confusing is because there is abbreviated 
# syntax, that works for python3-- specifically, the following is functionally 
# equivelnt;
#        super().__init__(data_points)
# ,,, which is matches what we saw in OOP python three book... and is clearer
# Anyway, the second part, .__init__(data_points), inherets the inialization code 
# from the parent cKDTree class
        self._processed     =   scipy.zeros((self.n,1),dtype=bool)
        self._reachability  =   scipy.ones(self.n)#*scipy.nan
        self._core_dist     =   scipy.ones(self.n)*scipy.nan
        self._index         =   scipy.array(range(self.n)
        self.nneighbor      =   scipy.ones(self.n,dtype=int)
# Alright, now we can do whatever we want. Note that processed is the added attribute
# that we wanted-- and it wasworking inspite of using processed earlier in the initailization 
# code, not* *because of it. Bad nameing and coindence, combinded with things not 
# breaking made it confusing...

# We set processed to false, reachability dist and core dist both to unknown
# (i.e., nan), and will use these to guild further processing..

#    def processed_filter(self):
#        self.processed = 'False'   #commented out?
#        self.n = n                  #commented out?

###working, butsaving fo later....
    def setneighborhood(self,point,epsilon):
        neighborhood     =  self.query(self.data[point],
                               self.n,
                               distance_upper_bound=epsilon)
    
        self.nneighbor[point] = len(scipy.isfinite(neighborhood[0]).nonzero()[0])
#  #      self.neighborhood[point] =  self.query(self.data[point],
#     #                          self.nneighbor,
#      #                         distance_upper_bound=epsilon)


    def __iter__(self):
        output = (self._index[n] for n in self._index if not self._processed[self._index[n]])

@autojit
def prep_optics(SetofObjects,epsilon):
    for i in range(SetofObjects.n):
        SetofObjects.setneighborhood(i,epsilon)
    print(str(SetofObjects.n) + ' points processed')


def build_optics(setOfObjects,epsilon,MinPts,Output_file_name):
    for point in xrange(setOfObjects.n):
        if setOfObjects._processed[point] == False:
            expandClusterOrder(setOfObjects,
#                               setOfObjects[point],epsilon, # pass point, or index?
                               point,epsilon, # pass point, or index?
                               MinPts,Output_file_name)
            
#Note captitaization of SetOfObjects to not call the build method
def expandClusterOrder(SetOfObjects,point,epsilon,MinPts,Output_file_name):
    
    neighbors = SetOfObjects.query(SetOfObjects.data[point],
                                   SetOfObjects.n,
                                   distance_upper_bound=epsilon)

 ##    sum(scipy.isfinite(test2[0]))  ### bad :(  , but equivlent
    num_neighbors = len(scipy.isfinite(neighbors[0]).nonzero()[0])
    SetOfObjects._processed[point] = True
    SetOfObjects._reachability[point] += scipy.nan
#    core_dist = SetOfObjects.query(SetOfObjects.data[point],MinPts,
    core_dist = SetOfObjects.query(SetOfObjects.data[point],MinPts,
#                                   distance_upper_bound=epsilon)[0][MinPts]
                                   distance_upper_bound=epsilon)[0][MinPts-1]
    SetOfObjects._core_dist[point]= core_dist
#    print core_dist
##Above; we want the last element, using MinPts is off by one

    if not scipy.isfinite(core_dist):
        SetOfObjects._core_dist[point] = scipy.nan
    with open(Output_file_name,'a') as file:
#        file.write((point,SetOfObjects._core_dist[point]))
#        file.write('tresting')
        file.write((str(point) + ' ' + str(SetOfObjects._core_dist[point]) + '\n'))
    if scipy.isfinite(core_dist):
#        orderSeeds_update(neighbors[1][:num_neighbors],point)
#above does't pass as it should...
        orderSeeds_update(neighbors[1][:num_neighbors],neighbors[0][:num_neighbors],point,SetOfObjects)
       
#        ###commented for now
#        
def orderSeeds_update(neighbors,distances,centerPoint,SetOfObjects):
    c_dist = setOfObjects._core_dist[centerPoint]
    new_r_dist = ((max(c_dist,point_dist) for point_dist in distances))        #neighbors[0]))
    
    for index in neighbors, point_dist in new_r_dist:
        if SetOfObjects._processed[index] == False:
            if scipy.isnan(SetOfObjects._reachability[index]):
                SetOfObjects._reachability[index] = point_dist 
                
    for d, i in neighbors[0],neighbors[1]:
        if setOfObjects._processed[i] == False:
            new_r_dist = max(c_dist,neighbors[0][i])
        
#

def return_unprocessed(mixed_points):

def set_reach_dist(SetOfObjects,point_index,epsilon):
    distances, indices = SetOfObjects.query(SetOfObjects.data[point_index],
                                            SetOfObjects._nneighbors[point_index],
                                            distance_upper_bound=epsilon)
    c_dist = distances[-1]
    unprocessed = SetOfPoints._index[(scipy.where(test_set._processed < 1)[0])]
    SetOfObjects._reachability[unprocessed] = scipy.minimum(SetOfPoints._reachability[unprocessed],cdist)
    return unprocessed[0]
    



###setting up testing data
testdata = scipy.rand(5000,3)
test_set = setOfObjects(testdata)
neighbors = test_set.query(test_set.data[1337],test_set.n,distance_upper_bound=.07)
num_neighbors = len(scipy.isfinite(neighbors[0]).nonzero()[0])
neighbors,distances=neighbors[1][:num_neighbors],neighbors[0][:num_neighbors]
core_dist = test_set.query(test_set.data[1337],num_neighbors,distance_upper_bound=.07)[0][num_neighbors-1]
centerPoint = 1337

####working####
#new_r_dist = ((max(c_dist,point_dist) for point_dist in distances))
#for index, point_dist in izip(neighbors,new_r_dist):
#    if test_set._processed[index] == False:
#        if scipy.isnan(test_set._reachability[index]):
#            test_set._reachability[index] = point_dist




