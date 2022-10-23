"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from data import make_dataset2
from plot import plot_boundary
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

  print("Accuracy for depth {} : {}" .format(depth, np.mean(acc)))
  print("Standard deviation for depth {} : {}" .format(depth, np.std(acc)))

if __name__ == "__main__":
  depths = [1,2,4,8,None]
  sample_size = 1500
  fit_size = 1200

  X, y = make_dataset2(sample_size)

  for depth in depths:
    test_depth(depth, sample_size, fit_size)

    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X[:fit_size], y[:fit_size])
    fname = "figs/dt/dt_{}" .format(depth)
    plot_boundary(fname, dt, X, y, title="Decision Tree with depth {}" .format(depth))

  
