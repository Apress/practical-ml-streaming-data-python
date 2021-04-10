#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 20:20:32 2020

@author: Dr. Sayan Putatunda
"""

""" Chapter 4 Codes """

import os
os.getcwd()  # to see the current path of working directory
os.chdir('./Python codes') 



###################################################



###################################################
# Mini-batch K-Means algorithm
from sklearn import datasets
from sklearn.cluster import MiniBatchKMeans

# Load data
iris = datasets.load_iris()
X = iris.data
#Data snapshot
X[0:6,]
#Output
#array([[5.1, 3.5, 1.4, 0.2],
#       [4.9, 3. , 1.4, 0.2],
#       [4.7, 3.2, 1.3, 0.2],
#       [4.6, 3.1, 1.5, 0.2],
#       [5. , 3.6, 1.4, 0.2],
#       [5.4, 3.9, 1.7, 0.4]])

#feature names
iris.feature_names
#Output:
#['sepal length (cm)',
# 'sepal width (cm)',
# 'petal length (cm)',
# 'petal width (cm)']

# Create k-mean object
clust_kmeans = MiniBatchKMeans(n_clusters=3, batch_size=50, random_state=333)

# Train model
model = clust_kmeans.fit(X)

# Figure out the cluster centers
model.cluster_centers_
#Output
#array([[5.97476415, 2.77051887, 4.48396226, 1.47122642],
#       [4.97687688, 3.37987988, 1.46696697, 0.24384384],
#       [6.87489712, 3.1       , 5.75226337, 2.1037037 ]])



##############################################################################################



# Anomaly detection- half-space trees
import pandas as pd
import numpy as np
from skmultiflow.anomaly_detection import HalfSpaceTrees
from skmultiflow.data import AnomalySineGenerator

# Generate a data stream with anomalies
dstream = AnomalySineGenerator(n_samples=500, n_anomalies=100, random_state=333)

# Instantiate the Half-Space Trees estimator
HS_trees = HalfSpaceTrees(random_state=333)

#prep the data stream
dstream.prepare_for_use()
X, Y = dstream.next_sample(500)
data = pd.DataFrame(np.hstack((X, np.array([Y]).T)))
data.head()

# Incrementally fit the model
HS_trees_model= HS_trees.partial_fit(X, Y)

# Predict the classes of the passed observations
pred = HS_trees_model.predict(X)

# Estimate the probability of a sample being anomalous or genuine
prob = HS_trees_model.predict_proba(X)

###################################################################################################




















