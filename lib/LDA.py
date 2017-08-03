#!/usr/bin/env python
"""
A module that contains the content-based recommender LDARecommender that uses
LDA.
"""
from lib.content_based import ContentBased
from overrides import overrides
from sklearn.decomposition import LatentDirichletAllocation


class LDARecommender(ContentBased):
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
        super(LDARecommender, self).__init__(initializer, evaluator, hyperparameters, options,
                                             verbose, load_matrices, dump_matrices)

    @overrides
    def train_one_fold(self, return_report=True):
        """
        Train one fold for n_iter iterations from scratch.

        :param bool return_report: A flag to decide if we should return the evaluation report.
        """
        # Try to read from file.
        matrix_found = False
        if self._load_matrices is True:
            matrix_shape = (self.n_items, self.n_topics)
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
        if return_report:
            return self.get_evaluation_report()

    def _train(self):
        """
        Train LDA Recommender, and store the document_distribution.
        """
        term_freq = self.abstracts_preprocessor.get_term_frequency_sparse_matrix()
        lda = LatentDirichletAllocation(n_topics=self.n_topics, max_iter=self.n_iter,
                                        learning_method='online',
                                        learning_offset=50., random_state=0,
                                        verbose=0)
        if self._verbose:
            print("Initialized LDA model..., Training LDA...")

        self.document_distribution = lda.fit_transform(term_freq)
        if self._verbose:
            print("LDA trained..")
