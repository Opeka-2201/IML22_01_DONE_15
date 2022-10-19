"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from data import make_dataset1, make_dataset2
from sklearn.neighbors import KNeighborsClassifier
from plot import plot_boundary

def test_neighbor(neighbor, n_samples, n_fit):
  n_tests = 5
  results = np.zeros(5)

  for i in range(n_tests):
    test = 0.
    data = make_dataset2(n_samples)
    data_points = data[0]
    data_colors = data[1]

    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X=data_points[:n_fit+1], y=data_colors[:n_fit+1])
    sample_colors = data_colors[n_fit+1:]
    predicted_colors = knn.predict(X=data_points[n_fit+1:])
    
    for j in range(len(sample_colors)):
      if sample_colors[j] == predicted_colors[j]:
        test+=1

    results[i] = test/(n_samples-n_fit)
  
  print("Results for n_neighbors = " + str(neighbor) + ":")
  print("Mean : " + str(results.mean()))
  print("Std : " + str(results.std()) + "\n")

if __name__ == "__main__":
  neighbors = [1,5,25,125,625,1200]
  sample_size = 1500
  fit_size = 1200
  data = make_dataset2(n_points=sample_size)
  data_points = data[0]
  data_colors = data[1]

  for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    fitted = knn.fit(X=data_points[:fit_size+1], y=data_colors[:fit_size+1])
    f_name = "figs/knn/knn_" + str(neighbor)
    f_title = "K-nearest neighbor with n_neighbors = " + str(neighbor)
    plot_boundary(f_name, fitted, data_points, data_colors, title=f_title)
    test_neighbor(neighbor, sample_size, fit_size)
