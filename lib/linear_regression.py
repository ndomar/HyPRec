#!/usr/bin/env python
"""
Module that trains a linear regression model to combine
content based and collaborative recommenders.
"""
import numpy
from sklearn import linear_model


class LinearRegression(object):
    """
    Linear regression to combine the results of two matrices.
    """
    def __init__(self, train_labels, test_labels, item_based_ratings, collaborative_ratings, training_data_mask):
        """
        Apply linear regression between two different methods to predict final collaborative_ratings

        :param ndarray train_labels: Training data.
        :param ndarray test_labels: Test data.
        :param ndarray item_based_ratings: Ratings produced by item based recommender.
        :param ndarray collaborative_ratings: Ratings produced by collaborative recommender
        :param ndarray training_data_mask: mask for the training data, dimensioin: users x items
        """
        self.item_based_ratings = item_based_ratings
        self.collaborative_ratings = collaborative_ratings
        self.item_based_ratings_shape = item_based_ratings.shape
        self.collaborative_ratings_shape = collaborative_ratings.shape


        # Mask the training data to get the ratings of only trianing data:
        item_based_ratings = self.flatten_matrix(item_based_ratings)[training_data_mask.flatten()]
        collaborative_ratings = self.flatten_matrix(collaborative_ratings)[training_data_mask.flatten()]

        # Build the training matrix for LR:
        self.train_data = numpy.vstack(( item_based_ratings,
                                         collaborative_ratings ) ).T

        # Mask the training data to get the labels of only trianing data:
        # self.flat_train_labels = self.flatten_matrix(train_labels)
        self.flat_train_labels = self.flatten_matrix(train_labels)[training_data_mask.flatten()]

        # Get teh test labels:
        self.flat_test_labels = self.flatten_matrix(test_labels)

        self.regression_coef1 = 0
        self.regression_coef2 = 0

    def flatten_matrix(self, matrix):
        """
        Method converts a matrix to a 1d array

        :param ndarray matrix: The matrix to be converted.
        :returns: flattened list
        :rtype: float[]
        """
        return matrix.flatten()

    def unflatten(self, matrix, shape):
        """
        Methods converts 1d array to a 2d array given a shape.

        :param float[] matrix: list to be converted.
        :param tuple(int) shape: Shape of the new matrix.

        :returns: 2D matrix
        :rtype: ndarray
        """
        return matrix.reshape(shape)

    def train(self):
        """
        Method trains a liner regression model

        :returns: adjusted predictions matrix.
        :rtype: ndarray
        """
        regr_model = linear_model.LinearRegression()
        regr_model.fit(self.train_data, self.flat_train_labels)
        weighted_item_based_ratings = regr_model.coef_[0] * self.item_based_ratings
        weighted_collaborative_ratings = regr_model.coef_[1] * self.collaborative_ratings
        self.regression_coef1 = regr_model.coef_[0]
        self.regression_coef2 = regr_model.coef_[1]
        print("CBF coef: {}".format(regr_model.coef_[0]))
        print("CF coef: {}".format(regr_model.coef_[1]))
        print("Intercept: {}".format(regr_model.intercept_))
        return weighted_collaborative_ratings + weighted_item_based_ratings +regr_model.intercept_
