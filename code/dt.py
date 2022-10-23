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
from plot import plot_boundary
from sklearn.tree import DecisionTreeClassifier, plot_tree

def test_depth(depth, n_samples, n_fit):
  n_tests = 5
  results = np.zeros(5)

  for i in range(n_tests):
    test = 0.
    data = make_dataset2(n_samples)
    data_points = data[0]
    data_colors = data[1]

    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X=data_points[:n_fit+1], y=data_colors[:n_fit+1])
    sample_colors = data_colors[n_fit+1:]
    predicted_colors = dt.predict(X=data_points[n_fit+1:])
    
    for j in range(len(sample_colors)):
      if sample_colors[j] == predicted_colors[j]:
        test+=1

    results[i] = test/(n_samples-n_fit)
  
  print("Results for depth = " + str(depth) + ":")
  print("Mean : " + str(results.mean()))
  print("Std : " + str(results.std()) + "\n")

if __name__ == "__main__":
  depths = [1,2,4,8,None]
  sample_size = 1500
  fit_size = 1200
  data = make_dataset2(n_points=sample_size)
  data_points = data[0]
  data_colors = data[1]

  for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth)
    fitted = dt.fit(X=data_points[:fit_size+1], y=data_colors[:fit_size+1])
    f_name = "figs/dt/dt_" + str(depth)
    tree_name = "figs/dt/tree_" + str(depth) + ".svg"
    f_title = "Decision tree with max_depth = " + str(depth)
    plot_boundary(f_name, fitted, data_points, data_colors, title=f_title)
    plot_tree(fitted, filled=True, feature_names=["x1", "x2"], class_names=["red", "blue"], rounded=True)
    plt.savefig(tree_name)
    test_depth(depth, sample_size, fit_size)
