"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from data import make_dataset2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from plot import plot_boundary

def test_neighbor(neighbor, n_samples, n_fit):
  n = 5
  acc = np.zeros(n)

  for i in range(n):
    X, y = make_dataset2(n_samples)
    X_fit, y_fit = X[:n_fit], y[:n_fit]
    X_test, y_test = X[n_fit:], y[n_fit:]

    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_fit, y_fit)
    y_pred = knn.predict(X_test)
    acc[i] = accuracy_score(y_test, y_pred)

  print("Accuracy for {} neighbors: {}" .format(neighbor, np.mean(acc)))
  print("Standard deviation for {} neighbors: {}" .format(neighbor, np.std(acc)))

if __name__ == "__main__":
  neighbors = [1,5,25,125,625,1200]
  sample_size = 1500
  fit_size = 1200

  X, y = make_dataset2(n_points=sample_size)

  for neighbor in neighbors:
    test_neighbor(neighbor, sample_size, fit_size)
      
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X[:fit_size], y[:fit_size])
    fname = "figs/knn/knn_{}" .format(neighbor)
    plot_boundary(fname, knn, X, y, title="KNN with {} neighbors" .format(neighbor))
