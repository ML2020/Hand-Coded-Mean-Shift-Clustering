
# Hand Coded Mean Shift Clustering code
#
# We can vary: our_radius to determine
# how our input data points X are to 
# clustered
#
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# our sample data, ordered into three 'clusterable'
# groups
#
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [8, 2],
              [10,2],
              [9, 3]])

# Note: because we have three generally clusterable 
# groups of data points defined above, if we use
# our_radius < 3, some groups of data points will
# directly get their own overlapping cluster point
#
# Likewise, a large our_radius, for example 30, will
# result in a single cluster data point
#
our_radius = 4

# scatter all the 0th and 1th elements in the array
#
#plt.scatter(X[:,0], X[:,1], s=150)
#plt.show()

colors = 10*["g","r","c","b","k"]


# Mean Shift algo steps:
# 1. assign every single feature set is a cluster center
#
# 2. then take all of the data pts or feature sets within each 
#    cluster's radius (or within the bandwidth) and take the 
#    mean of all those datasets or feature sets: that is our 
#    new cluster center
#
# 3. then repeat step 2 until you have convergence which
#   many of the clusters converge on each orther or stop
#   moving
# 
 

class Mean_Shift:
    def __init__(self, radius):
        self.radius = radius
#        print("init_radius: ", radius)

    def fit(self, data):
        centroids = {}
     
        for i in range(len(data)):
            centroids[i] = data[i]

# for Mean Shift you can have max_iterations and tolerance
# we'll have tolerance, but no max-iter
#
        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]

                for featureset in data:
# if the Euclidean distance is less than the bandwidth we're 
# allowing for: we are within the bandwidth / radius, so 
# append the featureset data
#
                    if (np.linalg.norm(featureset-centroid) < self.radius):
                        in_bandwidth.append(featureset)

# the following gives us the mean vector 
# of all of our vectors
#
                new_centroid = np.average(in_bandwidth, axis=0)

# then add this to a new centroids list, and 
# append the tuple version of the centroid
# ie. we're converting an array to a numpy tuple
#
                new_centroids.append(tuple(new_centroid))
    
# Now we want to get the unique elements of the centroids
# we used a tuple above since we can use set() on tuples
# unique of each value on the array, sorting the list
#
# As we get convergence, we'll get identicals, which
# we don't need, sort the rest
#
            uniques = sorted(list(set(new_centroids)))

# our way of copying the centroids dict without
# taking the attributes, w/o modifying prev centroids
# 
            prev_centroids = dict(centroids)
    
# define a new empty centroids dictionary
#
            centroids = {}
            
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

# assume optimized until break
#
            optimized = True

# now we check for any centroids movement:
# if we've found one that's moved, there's
# no reason to continue
#

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False

# if we've found one centroid that's moved, 
# we can break out
#
                if not optimized:
                    break

# break out of our while loop since if
# we are optimized:
#
            if optimized:
                break
        
#            if (i > (self.radius+1)):
#                    break

                    
# now we're optimized:
# where now outside the while True loop:
#
        self.centroids = centroids
    
    def predict(self, data):
        pass

# End of our MeanShift defn
#

clf = Mean_Shift(our_radius)

# fit our X data defined abobe using our self-defined
# Mean_Shift algo
#
clf.fit(X)

centroids = clf.centroids

# now plot our feature set data and centroids
#

# scatter plot our input data
#
plt.scatter(X[:,0], X[:,1], s=150)

# plot our centroids:
#
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color ='k', marker='*', s=150)
    print("c=: ", c, "centroids[c][0]: ", centroids[c][0], "centroids[c][1]: ", centroids[c][1],)
plt.show()

    
##################  End ################


"""

Output:

c=:  0 centroids[c][0]:  1.16666666667 centroids[c][1]:  1.46666666667
c=:  1 centroids[c][0]:  7.33333333333 centroids[c][1]:  9.0
c=:  2 centroids[c][0]:  9.0 centroids[c][1]:  2.33333333333


"""

