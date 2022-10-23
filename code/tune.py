"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from data import make_dataset2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

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
  return np.mean(acc), np.std(acc)

def test_depth(depth, n_samples, n_fit):
  n = 5
  acc = np.zeros(n)

  for i in range(n):
    X, y = make_dataset2(n_samples)
    X_fit, y_fit = X[:n_fit], y[:n_fit]
    X_test, y_test = X[n_fit:], y[n_fit:]

    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X_fit, y_fit)
    y_pred = dt.predict(X_test)
    acc[i] = accuracy_score(y_test, y_pred)
  return np.mean(acc), np.std(acc)


if __name__ == "__main__":
  neighbors = [1, 5, 25, 125, 625, 1200]
  sample_size = 1500
  fit_size = 1200
  tuning = np.zeros((len(neighbors), 3))

  X, y = make_dataset2(n_points=sample_size)
  i = 0
  for neighbor in neighbors:
    mean , std = test_neighbor(neighbor, sample_size, fit_size)
    tuning[i] = [neighbor, mean, std]
    i+=1
  bestnb = tuning[0,0]
  find = False

  for i in range(1,len(neighbors)):
    if tuning[i][1]/tuning[i-1][1] <= 1.02 and not find:
        bestnb = tuning[i-1][0]
        find = True
  print("the best number of neighbors is "+str(bestnb))

  depths = [1, 2, 4, 8, None]
  sample_size = 1500
  fit_size = 1200

  X, y = make_dataset2(sample_size)
  i = 0
  for depth in depths:
    mean, std = test_depth(depth, sample_size, fit_size)
    tuning[i] = [depth, mean, std]
    i += 1
  bestnb = tuning[0, 0]
  find = False
  for i in range(1, len(neighbors)):
    if tuning[i][1]/tuning[i-1][1] <= 1.02 and not find :
        bestnb = tuning[i-1][0]
        find = True
  print("the best depth is "+str(bestnb))
