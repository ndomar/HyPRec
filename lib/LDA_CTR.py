#!/usr/bin/env python
"""
A module that contains the content-based recommender LDARecommender that uses
LDA.
"""
from lib.content_based import ContentBased
from overrides import overrides
from sklearn.decomposition import LatentDirichletAllocation
import numpy
from numpy.linalg import solve


class LDACTRRecommender(ContentBased):
    """
    LDA Recommender, a content based recommender that uses LDA.
    """
    def __init__(self, initializer, evaluator, hyperparameters, options,
                 verbose=False, load_matrices=True, dump_matrices=True):
        """
        Constructor of Latent Dirichilet allocation's processor.

        :param ModelInitializer initializer: A model initializer.
        :param Evaluator evaluator: An evaluator of recommender and holder of input.
        :param dict hyperparameters: A dictionary of the hyperparameters.
        :param dict options: A dictionary of the run options.
        :param boolean verbose: A flag for printing while computing.
        :param boolean load_matrices: A flag for reinitializing the matrices.
        :param boolean dump_matrices: A flag for saving the matrices.
        """
        super(LDACTRRecommender, self).__init__(initializer, evaluator, hyperparameters, options,
                                             verbose, load_matrices, dump_matrices)

    def train_k_fold(self):
        all_errors = []
        for current_k in range(self.k_folds):
            self.set_data(*self.evaluator.get_fold(current_k, self.fold_test_indices))
            self.hyperparameters['fold'] = current_k
            self.train_one_fold(False)
            all_errors.append(self.get_evaluation_report())
        return numpy.mean(all_errors, axis=0)

    @overrides
    def train_one_fold(self, return_report=True):
        """
        Train one fold for n_iter iterations from scratch.

        :param bool return_report: A flag to decide if we should return the evaluation report.
        """
        # Try to read from file.
        matrix_found = False
        if self._load_matrices is True:
            matrix_shape = (self.n_items, self.n_factors)
            matrix_found, matrix = self.initializer.load_matrix(self.hyperparameters, 'document_distribution_lda',
                                                                matrix_shape)
            self.document_distribution = matrix
            if self._verbose and matrix_found:
                print("Document distribution was set from file, will not train.")
        if matrix_found is False:
            if self._verbose and self._load_matrices:
                print("Document distribution file was not found, will train LDA.")
            self._train()
            if self._dump_matrices:
                self.initializer.save_matrix(self.document_distribution, 'document_distribution_lda')
        self.user_vecs = numpy.random.random((self.n_users, self.n_factors))
        self.training_data_mask = self.evaluator.get_fold_training_data_mask()
        confidence_matrix = numpy.zeros(self.train_data.shape)
        confidence_matrix[self.training_data_mask] = self.train_data[self.training_data_mask]
        zeros_mask = self.train_data == 0
        training_zeros_mask = numpy.logical_and(self.training_data_mask, zeros_mask)
        confidence_matrix[training_zeros_mask] = 0.01
        self.user_vecs = self.als_step(self.user_vecs, self.document_distribution, self.train_data, self._lambda, confidence_matrix)
        if return_report:
            return self.get_evaluation_report()

    def _train(self):
        """
        Train LDA Recommender, and store the document_distribution.
        """
        term_freq = self.abstracts_preprocessor.get_term_frequency_sparse_matrix()
        lda = LatentDirichletAllocation(n_topics=self.n_factors, max_iter=self.n_iter,
                                        learning_method='online',
                                        learning_offset=50., random_state=0,
                                        verbose=0)
        if self._verbose:
            print("Initialized LDA model..., Training LDA...")

        self.document_distribution = lda.fit_transform(term_freq)
        if self._verbose:
            print("LDA trained..")
    

    def als_step(self, latent_vectors, fixed_vecs, ratings, _lambda, confidence_matrix):
        """
        The function computes only one step in the ALS algorithm

        :param ndarray latent_vectors: the vector to be optimized
        :param ndarray fixed_vecs: the vector to be fixed
        :param ndarray ratings: ratings that will be used to optimize latent * fixed
        :param float _lambda: reguralization parameter
        """
        numpy.where(">1")
        lambdaI = numpy.eye(self.n_factors) * _lambda
        for u in range(latent_vectors.shape[0]):
            """
            Omar code:
            confidence = self.build_confidence_matrix(u, 'user')
            YTY = (fixed_vecs.T * confidence).dot(fixed_vecs)
            latent_vectors[u, :] = solve((YTY + lambdaI), (ratings[u, :] * confidence).dot(fixed_vecs))
            """
            # Anas Code
            YTY = (fixed_vecs.T * confidence_matrix[u]).dot(fixed_vecs)
        return latent_vectors

    def get_predictions(self):
        """
        Get the expected ratings between users and items.

        :returns: A matrix of users X documents
        :rtype: ndarray
        """
        if self.predictions is None:
            self.predictions = self.user_vecs.dot(self.document_distribution.T)
        return self.predictions
