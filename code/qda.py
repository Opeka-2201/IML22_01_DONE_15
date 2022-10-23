"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from data import make_dataset1, make_dataset2
from plot import plot_boundary

class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.pi_ = np.zeros(2, dtype=np.float64)
        self.means_ = np.zeros((2,2), dtype=np.float64)
        self.cov_ = np.zeros((2,2,2), dtype=np.float64)
        self.lda = False

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

        self.means_[0][0] = np.mean(X_0[0])
        self.means_[0][1] = np.mean(X_0[1])
        self.means_[1][0] = np.mean(X_1[0])
        self.means_[1][1] = np.mean(X_1[1])

        if not(lda):
            self.cov_[0] = np.cov(X_0.T)
            self.cov_[1] = np.cov(X_1.T)
        else:
            self.cov_[0] = np.cov(X.T)
            self.cov_[1] = np.cov(X.T)

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

        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

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

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        n_samples = X.shape[0]
        n_classes = len(self.pi_)

        probabilities = np.zeros((n_samples, n_classes), dtype=np.float64)

        for i in range(n_classes):
            probabilities[:, i] = self.pi_[i] * self._multivariate_normal(X, self.means_[i], self.cov_[i])

        # Normalize probabilities
        probabilities /= np.sum(probabilities, axis=1)[:, np.newaxis]

        return probabilities

    def _multivariate_normal(self, X, mean, cov):
        """Return the multivariate normal distribution.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        mean : array-like of shape = [n_features]
            The mean of the distribution.

        cov : array-like of shape = [n_features, n_features]
            The covariance matrix of the distribution.

        Returns
        -------
        p : array of shape = [n_samples]
            The probability of each sample.
        """

        n_samples = X.shape[0]

        probabilities = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            probabilities[i] = 1 / np.sqrt(np.linalg.det(2 * np.pi * cov)) * np.exp(-0.5 * np.dot(np.dot((X[i] - mean), np.linalg.inv(cov)), (X[i] - mean).T))

        return probabilities

def test_qda(n_samples, n_fit):
    n_tests = 5

    accuracy_qda1 = np.zeros(n_tests)
    accuracy_lda1 = np.zeros(n_tests)
    accuracy_qda2 = np.zeros(n_tests)
    accuracy_lda2 = np.zeros(n_tests)

    for i in range(n_tests):
        X1, y1 = make_dataset1(n_samples)
        X2, y2 = make_dataset2(n_samples)

        qda1 = QuadraticDiscriminantAnalysis()
        qda1.fit(X1[:n_fit], y1[:n_fit])
        accuracy_qda1[i] = accuracy_score(y1[n_fit:], qda1.predict(X1[n_fit:]))

        lda1 = QuadraticDiscriminantAnalysis()
        lda1.fit(X1[:n_fit], y1[:n_fit], lda=True)
        accuracy_lda1[i] = accuracy_score(y1[n_fit:], lda1.predict(X1[n_fit:]))

        qda2 = QuadraticDiscriminantAnalysis()
        qda2.fit(X2[:n_fit], y2[:n_fit])
        accuracy_qda2[i] = accuracy_score(y2[n_fit:], qda2.predict(X2[n_fit:]))

        lda2 = QuadraticDiscriminantAnalysis()
        lda2.fit(X2[:n_fit], y2[:n_fit], lda=True)
        accuracy_lda2[i] = accuracy_score(y2[n_fit:], lda2.predict(X2[n_fit:]))

    print("Average accuracy of QDA on dataset 1: %f" % np.mean(accuracy_qda1))
    print("Average accuracy of LDA on dataset 1: %f" % np.mean(accuracy_lda1))
    print("Average accuracy of QDA on dataset 2: %f" % np.mean(accuracy_qda2))
    print("Average accuracy of LDA on dataset 2: %f" % np.mean(accuracy_lda2))

    print("Standard deviation of QDA on dataset 1: %f" % np.std(accuracy_qda1))
    print("Standard deviation of LDA on dataset 1: %f" % np.std(accuracy_lda1))
    print("Standard deviation of QDA on dataset 2: %f" % np.std(accuracy_qda2))
    print("Standard deviation of LDA on dataset 2: %f" % np.std(accuracy_lda2))


if __name__ == "__main__":
    n_samples = 1500
    n_fit = 1200

    X, y = make_dataset2(n_points=1500)

    qda_2 = QuadraticDiscriminantAnalysis()
    lda_2 = QuadraticDiscriminantAnalysis()

    qda_2.fit(X[:n_fit+1], y[:n_fit+1], lda=False)
    lda_2.fit(X[:n_fit+1], y[:n_fit+1], lda=True)

    qda_2_name="figs/qda/qda2"
    lda_2_name="figs/qda/lda2"

    plot_boundary(qda_2_name, qda_2, X, y, title="QDA on dataset 2")
    plot_boundary(lda_2_name, lda_2, X, y, title="LDA on dataset 2")

    test_qda(n_samples, n_fit)