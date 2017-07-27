#!/usr/bin/env python
"""
Module that provides the main functionalities of collaborative filtering.
"""
import time
import numpy
from numpy.linalg import solve
from overrides import overrides
from lib.abstract_recommender import AbstractRecommender
from lib.linear_regression import LinearRegression


class CollaborativeFiltering(AbstractRecommender):
    """
    A class that takes in the rating matrix and outputs user and item
    representation in latent space.
    """
    def __init__(self, initializer, evaluator, hyperparameters, options,
                 verbose=False, load_matrices=True, dump_matrices=True, train_more=True,
                 is_hybrid=False, update_with_items=False, init_with_content=True):
        """
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a ratings matrix which is ~ user x item

        :param ModelInitializer initializer: A model initializer.
        :param Evaluator evaluator: Evaluator of the recommender and holder of the input data.
        :param dict hyperparameters: hyperparameters of the recommender, contains _lambda and n_factors
        :param dict options: Dictionary of the run options, contains n_iterations and k_folds
        :param boolean verbose: A flag if True, tracing will be printed
        :param boolean load_matrices: A flag for reinitializing the matrices.
        :param boolean dump_matrices: A flag for saving the matrices.
        :param boolean train_more: train_more the collaborative filtering after loading matrices.
        :param boolean is_hybrid: A flag indicating whether the recommender is hybrid or not.
        :param boolean update_with_items: A flag the decides if we will use the items matrix in the update rule.
        """
        # setting input
        self.initializer = initializer
        self.evaluator = evaluator
        self.ratings = evaluator.get_ratings()
        self.n_users, self.n_items = self.ratings.shape
        self.k_folds = None
        self.prediction_fold = -1

        # setting flags
        self._verbose = verbose
        self._load_matrices = load_matrices
        self._dump_matrices = dump_matrices
        self._train_more = train_more
        self._is_hybrid = is_hybrid
        self._update_with_items = update_with_items
        self._split_type = 'user'
        self._init_with_content = init_with_content

        self.set_hyperparameters(hyperparameters)
        self.set_options(options)
        self.document_distribution = None

    @overrides
    def set_hyperparameters(self, hyperparameters):
        """
        The function sets the hyperparameters of the uv_decomposition algorithm

        :param dict hyperparameters: hyperparameters of the recommender, contains _lambda and n_factors
        """
        self.n_factors = hyperparameters['n_factors']
        self._lambda = hyperparameters['_lambda']
        self.predictions = None
        self.hyperparameters = hyperparameters.copy()

    def als_step(self, latent_vectors, fixed_vecs, ratings, _lambda, confidence_matrix, type='user'):
        """
        The function computes only one step in the ALS algorithm

        :param ndarray latent_vectors: the vector to be optimized
        :param ndarray fixed_vecs: the vector to be fixed
        :param ndarray ratings: ratings that will be used to optimize latent * fixed
        :param float _lambda: reguralization parameter
        :param str type: either user or item.
        """
        if type == 'user':
            # Precompute
            numpy.where(">1")
            lambdaI = numpy.eye(self.hyperparameters['n_factors']) * _lambda
            for u in range(latent_vectors.shape[0]):
                """
                Omar code:
                confidence = self.build_confidence_matrix(u, 'user')
                YTY = (fixed_vecs.T * confidence).dot(fixed_vecs)
                latent_vectors[u, :] = solve((YTY + lambdaI), (ratings[u, :] * confidence).dot(fixed_vecs))
                """
                # Anas Code
                YTY = (fixed_vecs.T * confidence_matrix[u]).dot(fixed_vecs)
                latent_vectors[u, :] = solve((YTY + lambdaI), (ratings[u, :] * confidence_matrix[u]).dot(fixed_vecs))

        elif type == 'item':
            # Precompute
            lambdaI = numpy.eye(self.hyperparameters['n_factors']) * _lambda
            for i in range(latent_vectors.shape[0]):
                """
                Omar code:
                confidence = self.build_confidence_matrix(i, 'item')
                XTX = (fixed_vecs.T * confidence).dot(fixed_vecs)
                #if self.document_distribution is None and self._update_with_items:
                #    print("Update with items is true, but Document distribution is none at als_step")
                """
                # Anas Code
                XTX = (fixed_vecs.T * confidence_matrix[:,i]).dot(fixed_vecs)
                if self._update_with_items and self.document_distribution is not None:
                    #print("Update with items is true, and Document distribution is not none at als_step")
                    latent_vectors[i, :] = solve((XTX + lambdaI), (ratings[:, i].T * confidence_matrix[:,i]).dot(fixed_vecs) + self.document_distribution[i, :] * _lambda)
                else:
                    latent_vectors[i, :] = solve((XTX + lambdaI), (ratings[:, i].T * confidence_matrix[:,i]).dot(fixed_vecs))
        return latent_vectors

    @overrides
    def train(self, item_vecs=None):
        """
        Train the collaborative filtering.

        :param ndarray item_vecs: optional initalization for the item_vecs matrix.
        """
        if item_vecs is not None:
            self.document_distribution = item_vecs.copy()
        if self.splitting_method == 'naive':
            self.set_data(*self.evaluator.naive_split(self._split_type))
            self.hyperparameters['fold'] = 0
            return self.train_one_fold(item_vecs)
        else:
            self.fold_test_indices = self.evaluator.get_kfold_indices()
            return self.train_k_fold(item_vecs)

    def build_confidence_matrix(self, index, type='user'):
        """
        Builds a confidence matrix

        :param int index: Index of the user or item to build confidence for.
        :param str type: Type of confidence matrix, either user or item.

        :returns: A confidence matrix
        :rtype: ndarray
        """
        if type == 'user':
            shape = self.item_vecs.shape[0]
        else:
            shape = self.user_vecs.shape[0]

        confidence = numpy.array([0.1] * shape)
        for i in range(len(confidence)):
            if type == 'user':
                if not self.train_data[index][i] == 0:
                    confidence[i] = 1
            else:
                if not self.train_data[i][index] == 0:
                    confidence[i] = 1

        return confidence

    @overrides
    def train_k_fold(self, item_vecs=None):
        """
        Trains k folds of collaborative filtering.

        :returns: List of error metrics.
        :rtype: list[float]
        """
        all_errors = []
        for current_k in range(self.k_folds):
            self.set_data(*self.evaluator.get_fold(current_k, self.fold_test_indices))

            # Get the training data mask
            self.training_data_mask = self.evaluator.get_fold_training_data_mask()

            self.hyperparameters['fold'] = current_k
            current_error = self.train_one_fold(item_vecs)
            all_errors.append(current_error)
            self.predictions = None
        return numpy.mean(all_errors, axis=0)

    @overrides
    def train_one_fold(self, item_vecs=None):
        """
        Train one fold for n_iter iterations from scratch.

        :returns: List of error metrics.
        :rtype: list[float]
        """
        matrices_found = False
        if self._load_matrices is False:
            self.user_vecs = numpy.random.random((self.n_users, self.n_factors))

            # If init_with_content, then don't initialize V randomly, but with the content
            if (item_vecs is None or not self._init_with_content
                    or not item_vecs.shape == (self.n_items, self.n_factors)):
                self.item_vecs = numpy.random.random((self.n_items, self.n_factors))
                print("Init_with_content is false, V is initialized randomly")
            else:
                self.item_vecs = item_vecs.copy()
                print("Init_with_content is true, V is initialized with Document distributions")
        else:
            if self._verbose:
                print("Loading users and items matrices...")
            users_found, self.user_vecs = self.initializer.load_matrix(self.hyperparameters,
                                                                       'user_vecs' + self._get_options_suffix(),
                                                                       (self.n_users, self.n_factors))
            if self._verbose and users_found:
                print("Users matrix files were found and loaded")
            if self._verbose and not users_found:
                print("Users matrix files were not found, initialized randomly")

            items_found, self.item_vecs = self.initializer.load_matrix(self.hyperparameters,
                                                                       'item_vecs' + self._get_options_suffix(),
                                                                       (self.n_items, self.n_factors))
            if self._verbose:
                if items_found:
                    print("Items matrix files were found and loaded")
                else:
                    print("Items matrix files were not found, initialized randomly")

            if not items_found:
                # If init_with_content, and the items matrix file is not found then initialize V with the content
                if (item_vecs is not None and self._init_with_content):
                    self.item_vecs = item_vecs.copy()
                    if self._verbose:
                        print("Init_with_content is true, V is initialized with Document distributions")
            matrices_found = users_found and items_found
        if not matrices_found:
            if self._verbose and self._load_matrices:
                print("User and Document distributions files were not found, will train collaborative.")
            self.partial_train()
        else:
            if self._train_more:
                if self._verbose and self._load_matrices:
                    print("User and Document distributions files found, will train model further.")
                self.partial_train()
            else:
                if self._verbose and self._load_matrices:
                    print("User and Document distributions files found, will not train the model further.")
        """
        if self._dump_matrices:
            self.initializer.set_config(self.hyperparameters, self.n_iter)
            self.initializer.save_matrix(self.user_vecs, 'user_vecs' + self._get_options_suffix())
            self.initializer.save_matrix(self.item_vecs, 'item_vecs' + self._get_options_suffix())
        """
        return self.get_evaluation_report()

    def _get_options_suffix(self):
        suffix = ''
        if self._init_with_content:
            suffix += 'i'
        if self._update_with_items:
            suffix += 'u'
        if suffix:
            suffix = '_' + suffix
        return suffix

    def partial_train(self):
        """
        Train model for n_iter iterations. Can be called multiple times for further training.
        """
        error = numpy.inf
        if 'fold' in self.hyperparameters:
            current_fold = self.hyperparameters['fold'] + 1
        else:
            current_fold = 0
        if self._verbose:
            error = self.evaluator.get_rmse(self.user_vecs.dot(self.item_vecs.T), self.train_data)
            if current_fold == 0:
                print('Epoch:{epoch:02d} Loss:{loss:1.4e} Time:{time:.3f} s'.format(**dict(epoch=0, loss=error,
                                                                                          time=0)))
            else:
                print('Fold:{fold:02d} Epoch:{epoch:02d} Loss:{loss:1.4e} '
                      'Time:{time:.3f} s'.format(**dict(fold=current_fold, epoch=0, loss=error, time=0)))
        """
        Added by Anas (creating the TRAINING Confidence matrix:
        n x m matrix with:
        0.01 for the unknown ratings in the unobserved ratings from the training matrix,
        1 at the observed ratings, and
        0 for all test ratings (observed and unobserved)
        """
        confidence_matrix = numpy.zeros(self.train_data.shape)
        confidence_matrix[self.training_data_mask] = self.train_data[self.training_data_mask]
        zeros_mask = self.train_data == 0
        training_zeros_mask = numpy.logical_and(self.training_data_mask, zeros_mask)
        confidence_matrix[training_zeros_mask] = 0.01

        for epoch in range(1, self.n_iter + 1):
            t0 = time.time()
            old_error = error
            self.user_vecs = self.als_step(self.user_vecs, self.item_vecs, self.train_data, self._lambda, confidence_matrix, type='user')
            self.item_vecs = self.als_step(self.item_vecs, self.user_vecs, self.train_data, self._lambda, confidence_matrix, type='item')
            t1 = time.time()
            error = self.evaluator.get_rmse(self.user_vecs.dot(self.item_vecs.T), self.train_data)
            if self._verbose:
                if current_fold == 0:
                    print('Epoch:{epoch:02d} Loss:{loss:1.4e} Time:{time:.3f} s'.format(**dict(epoch=epoch, loss=error,
                                                                                              time=(t1 - t0))))
                else:
                    print('Fold:{fold:02d} Epoch:{epoch:02d} Loss:{loss:1.4e} '
                          'Time:{time:.3f} s'.format(**dict(fold=current_fold, epoch=epoch, loss=error,
                                                           time=(t1 - t0))))
            if error >= old_error:
                if self._verbose:
                    print("Local Optimum was found in the last iteration, breaking.")
                break

    @overrides
    def get_predictions(self):
        """
        Predict ratings for every user and item.

        :returns: A (user, document) matrix of predictions
        :rtype: ndarray
        """
        if self.predictions is None or not self.prediction_fold == self.hyperparameters['fold']:
            collaborative_predictions = self.user_vecs.dot(self.item_vecs.T)
            self.prediction_fold = self.hyperparameters['fold']

            self.predictions = collaborative_predictions

        return self.predictions

    @overrides
    def predict(self, user, item):
        """
        Single user and item prediction.

        :returns: prediction score
        :rtype: float
        """
        return self.user_vecs[user, :].dot(self.item_vecs[item, :].T)

    def set_item_based_recommender(self, recommender):
        """
        Set the item_based recommender, in order to use it as a hybrid recommender.

        :param ContentBased recommender: The content based recommender
        """
        self.item_based_recommender = recommender
