"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from data import make_dataset1, make_dataset2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def tune_knn(X_train, y_train):
    """
    Tune the hyperparameters of the KNN classifier
    """

    param_grid = {'n_neighbors': [1, 5, 25, 125, 625]}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def tune_decision_tree(X_train, y_train):
    """
    Tune the hyperparameters of the decision tree classifier
    """

    param_grid = {'max_depth': [1, 2, 4, 8, None]}
    tree = DecisionTreeClassifier()
    grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


if __name__ == '__main__':
    n_samples = 1500
    n_fit = 1200
    n_tests = 5
    results_1 = np.zeros((n_tests, 2))
    results_2 = np.zeros((n_tests, 2))

    for i in range(n_tests):
        n_samples = 1500
        n_fit = 1200
        
        X_1, y_1 = make_dataset1(n_samples)
        X_1_train, y_1_train = X_1[:n_fit], y_1[:n_fit]
        X_1_test, y_1_test = X_1[n_fit:], y_1[n_fit:]

        X_2, y_2 = make_dataset2(n_samples)
        X_2_train, y_2_train = X_2[:n_fit], y_2[:n_fit]
        X_2_test, y_2_test = X_2[n_fit:], y_2[n_fit:]

        knn1, params = tune_knn(X_1_train, y_1_train)
        results_1[i, 0] = params['n_neighbors']

        knn2, params = tune_knn(X_2_train, y_2_train)
        results_2[i, 0] = params['n_neighbors']

        tree1, params = tune_decision_tree(X_1_train, y_1_train)
        results_1[i, 1] = params['max_depth']

        tree2, params = tune_decision_tree(X_2_train, y_2_train)
        results_2[i, 1] = params['max_depth']

    best_knn_1 = np.bincount(results_1[:, 0].astype(int)).argmax()
    best_tree_1 = np.bincount(results_1[:, 1].astype(int)).argmax()
    best_knn_2 = np.bincount(results_2[:, 0].astype(int)).argmax()
    best_tree_2 = np.bincount(results_2[:, 1].astype(int)).argmax()

    print('Best hyperparameters for dataset 1:')
    print('KNN 1: {}'.format(best_knn_1))
    print('Decision tree 1: {}'.format(best_tree_1))

    print('Best hyperparameters for dataset 2:')
    print('KNN 2: {}'.format(best_knn_2))
    print('Decision tree 2: {}'.format(best_tree_2))

    results_1 = np.zeros((n_tests, 2))
    results_2 = np.zeros((n_tests, 2))

    for i in range(n_tests):
        X_1, y_1 = make_dataset1(n_samples)
        X_1_train, y_1_train = X_1[:n_fit], y_1[:n_fit]
        X_1_test, y_1_test = X_1[n_fit:], y_1[n_fit:]

        X_2, y_2 = make_dataset2(n_samples)
        X_2_train, y_2_train = X_2[:n_fit], y_2[:n_fit]
        X_2_test, y_2_test = X_2[n_fit:], y_2[n_fit:]

        knn1 = KNeighborsClassifier(n_neighbors=best_knn_1)
        knn1.fit(X_1_train, y_1_train)
        results_1[i, 0] = knn1.score(X_1_test, y_1_test)

        knn2 = KNeighborsClassifier(n_neighbors=best_knn_2)
        knn2.fit(X_2_train, y_2_train)
        results_2[i, 0] = knn2.score(X_2_test, y_2_test)

        tree1 = DecisionTreeClassifier(max_depth=best_tree_1)
        tree1.fit(X_1_train, y_1_train)
        results_1[i, 1] = tree1.score(X_1_test, y_1_test)

        tree2 = DecisionTreeClassifier(max_depth=best_tree_2)
        tree2.fit(X_2_train, y_2_train)
        results_2[i, 1] = tree2.score(X_2_test, y_2_test)

    print("Accuracy with best hyperparameters for dataset 1:")
    print('    KNN 1: {}'.format(results_1[:, 0].mean()))
    print('    Decision tree 1: {}'.format(results_1[:, 1].mean()))
    print("Standard deviation with best hyperparameters for dataset 1:")
    print('    KNN 1: {}'.format(results_1[:, 0].std()))
    print('    Decision tree 1:  {}'.format(results_1[:, 1].std()))

    print("Accuracy with best hyperparameters for dataset 2:")
    print('    KNN 2: {}'.format(results_2[:, 0].mean()))
    print('    Decision tree 2: {}'.format(results_2[:, 1].mean()))
    print("Standard deviation with best hyperparameters for dataset 2:")
    print('    KNN 2: {}'.format(results_2[:, 0].std()))
    print('    Decision tree 2:  {}'.format(results_2[:, 1].std()))