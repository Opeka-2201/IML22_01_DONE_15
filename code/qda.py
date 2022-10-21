"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.pi_ = np.zeros(2, dtype=np.float64)
        self.means_ = np.zeros((2, 2), dtype=np.float64)
        self.cov_0 = np.array((2, 2), dtype=np.float64)
        self.cov_1 = np.array((2, 2), dtype=np.float64)

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

        X_0 = X[y == 0]        
        X_1 = X[y == 1]

        self.pi_[0] = len(X_0) / len(X)
        self.pi_[1] = len(X_1) / len(X)

        print(self.pi_)

        self.means_[0] = np.mean(X_0)
        self.means_[1] = np.mean(X_1)

        if lda:
            self.cov_0 = np.cov(np.transpose(X_0))
            self.cov_1 = self.cov_0
        else:
            self.cov_0 = np.cov(np.transpose(X_0))
            self.cov_1 = np.cov(np.transpose(X_1))

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
        pred = self.predict_proba(X)
        test = np.zeros(len(X))
        for i in range(len(X)):
          test[i] = 0 if pred[i][0] > pred[i][1] else 1
        return test
        

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
        coef_0 = 1/(2*np.pi*np.sqrt(np.linalg.det(self.cov_0)))
        P_y = np.zeros((X.shape[0],2),dtype=np.float64) 
        for i in range(X.shape[0]):
          f_k = coef_0 * np.exp(-0.5 * np.transpose(X[i] - self.means_[0]) * np.linalg.inv(self.cov_0)*(X[i] - self.means_[0]))
          print(f_k)
          P_y[i][0] = (f_k*self.means_[0])/(f_k*self.pi_[0] + (1-f_k)*self.pi_[1])
          P_y[i][1] = 1 - P_y[i][0]
        return P_y


if __name__ == "__main__":
    from data import make_dataset1
    from plot import plot_boundary
    n_samples = 1500
    n_fit = 1200
    data = make_dataset1(n_points=1500)
    X = data[0]
    y = data[1]
    qda = QuadraticDiscriminantAnalysis()
    lda = QuadraticDiscriminantAnalysis()
    fitted_qda = qda.fit(X=X[:n_fit+1], y=y[:n_fit+1], lda=False)
    fitted_lda = lda.fit(X=X[:n_fit+1], y=y[:n_fit+1], lda=True)

    #print(fitted_qda.predict(X[n_fit+1:]))
    #print(fitted_lda.predict(X[n_fit+1:]))

    qda_name="/figs/qda"
    lda_name="/figs/lda"

    plot_boundary(qda_name, fitted_qda, X, y, title="QDA")
    plot_boundary(qda_name, fitted_lda, X, y, title="QDA")
