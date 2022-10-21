"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

from statistics import covariance
from data import make_dataset1
from plot import plot_boundary
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin



class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):

    covariance_matrix_1 = np.array([])
    covariance_matrix_2 = np.array([])
    classes = []
    mu = []
    pi = []

    def fit(self, X, y, lda=False):
        """Fit a linear discriminant analysis model using the training set
        (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        self.lda = lda

        samples = np.hstack((X, np.asarray([y]).T))

        dict = {}

        for sample in samples:
            if sample[-1] in dict:
              dict[sample[-1]][0] += 1
              dict[sample[-1]][1] = np.add(dict[sample[-1]][1], sample[0:-1])
            else:
              dict[sample[-1]] = [1, np.array(sample[0:-1])]

        for key in dict:
            dict[key][1] = np.divide(dict[key][1], dict[key][0])
            dict[key][0] = dict[key][0]/y.shape[0]

            dict_sort = sorted(dict.items(), key = lambda x:x[0])

            for entry in dict_sort:
              self.classes.append(entry[0])
              self.pi.append(entry[1][0])
              self.mu.append(entry[1][1])
            
            if self.lda:
              self.covariance_matrix_1 = np.cov(X, rowvar=False)
            else:
              pass

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        probas = self.predict_proba(X)
        y = np.zeros(np.asarray(X).shape[0])
        for i, _ in enumerate(X,0):
          y[i] = self.classes[np.argmax(probas[i])]

        return y

    def gauss(self, k, x):
      return np.divide(np.exp(-1/2*(x-self.mu[self.classes.index(k)])@np.linalg.inv(self.covariance_matrix_1)@(x-self.mu[self.classes.index(k)]).T), 
                        np.power(2*np.pi,x.ndim/2)*np.sqrt(np.linalg.det(self.covariance_matrix_1)))

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        probas = np.zeros([np.asarray(X).shape[0], np.asarray(self.classes).shape[0]])

        for i, x in enumerate(X,0):
          for j, k in enumerate(self.classes, 0):
            sum = 0
            for l in self.classes:
              sum += self.gauss(self.classes.index(l), x) * self.pi[self.classes.index(l)]

            probas[i,j] = self.gauss(self.classes.index(k), x) * self.pi[self.classes.index(l)] / sum

        return probas

if __name__ == "__main__":
  sample_size = 1500
  fit_size = 1200
  data = make_dataset1(n_points=sample_size)
  data_points = data[0]
  data_colors = data[1]

  lda = QuadraticDiscriminantAnalysis()
  fitted = lda.fit(X=data_points[:fit_size+1], y=data_colors[:fit_size+1], lda=True)
  f_name = "figs/lda"
  f_title = "LDA"
  plot_boundary(f_name, fitted, data_points, data_colors, title=f_title)