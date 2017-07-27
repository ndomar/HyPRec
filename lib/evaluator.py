#!/usr/bin/env python
"""
A module that provides functionalities for calculating error metrics
and evaluates the given recommender.
"""
import numpy
from util.top_recommendations import TopRecommendations


class Evaluator(object):
    """
    A class for computing evaluation metrics and splitting the input data.
    """
    def __init__(self, ratings, abstracts_preprocessor=None, random_seed=False,
                 verbose=False):
        """
        Initialize an evaluator array with the initial actual ratings matrix.

        :param int[][] ratings: A numpy array containing the initial ratings.
        :param AbstractsPreprocessor abstracts_preprocessor: A list of the abstracts.
        :param bool random_seed: if False, we will use a fixed seed.
        :param bool verbose: A flag deciding to print progress
        """
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        if abstracts_preprocessor:
            self.abstracts_preprocessor = abstracts_preprocessor
        self.random_seed = random_seed
        self._verbose = verbose
        self.k_folds = None

        if self._verbose:
            print('%d users and %d items' % (self.n_users, self.n_items))

        # stores recommended indices for each user.
        self.recommendation_indices = [[] for i in range(self.ratings.shape[0])]
        # False if recommendations have not been loaded yet and vice versa.
        self.recs_loaded = False

    def get_abstracts_preprocessor(self):
        """
        Getter for the Abstracts preprocessor.

        :returns: abstracts preprocessor
        :rtype: AbstractsPreprocessor
        """
        return self.abstracts_preprocessor

    def get_ratings(self):
        """
        Getter for the ratings matrix.

        :returns: Ratings matrix
        :rtype: ndarray
        """
        return self.ratings

    def set_kfolds(self, kfolds):
        """
        Set the k-folds

        :param int kfolds: the number of the folds in K-fold
        """
        self.k_folds = kfolds
        self.test_percentage = 1.0 / self.k_folds

    def naive_split(self, type='user'):
        """
        Split the data into training and testing sets.

        :returns: a tuple of train and test data.
        :rtype: tuple
        """
        if type == 'user':
            return self.naive_split_users()
        return self.naive_split_items()

    def naive_split_users(self):
        """
        Split the ratings into test and train data for every user.

        :returns: a tuple of train and test data.
        :rtype: tuple
        """
        if self.random_seed is False:
            numpy.random.seed(42)

        test = numpy.zeros(self.ratings.shape)
        train = self.ratings.copy()
        for user in range(self.ratings.shape[0]):
            non_zeros = self.ratings[user, :].nonzero()[0]
            test_ratings = numpy.random.choice(non_zeros,
                                               size=int(self.test_percentage * len(non_zeros)))
            train[user, test_ratings] = 0.
            test[user, test_ratings] = self.ratings[user, test_ratings]
        assert(numpy.all((train * test) == 0))
        self.test_indices = test
        return train, test

    def naive_split_items(self):
        """
        Split the ratings on test and train data by removing random documents.

        :returns: a tuple of train and test data.
        :rtype: tuple
        """
        if self.random_seed is False:
            numpy.random.seed(42)

        indices = list(range(self.n_items))
        test_ratings = numpy.random.choice(indices, size=int(self.test_percentage * len(indices)))
        train = self.ratings.copy()
        test = numpy.zeros(self.ratings.shape)
        for index in test_ratings:
            train[:, index] = 0
            test[:, index] = self.ratings[:, index]
        assert(numpy.all((train * test) == 0))
        return train, test

    def get_fold(self, fold_num, fold_test_indices):
        """
        Returns train and test data for a given fold number

        :param int fold_num: the fold index to be returned
        :param int[] fold_test_indices: A list of the indicies of the testing fold.
        :returns: tuple of training and test data
        :rtype: 2-tuple of 2d numpy arrays
        """
        self.current_test_fold_indices = []
        index = fold_num
        for ctr in range(self.ratings.shape[0]):
            self.current_test_fold_indices.append(fold_test_indices[index])
            index += self.k_folds
        return self.generate_kfold_matrix(self.current_test_fold_indices)

    def get_fold_training_data_mask(self):
        """
        Returns training data mask for the current fold, can be invoked only after evaluator.get_fold().

        :returns: 2d array of size users x items, with true values corresponding to trainging ratings
        :rtype: 2d numpy array
        """

        # the 2d array fold_train_indices_mask is already filled for the curent fold by generate_kfold_matrix
        return numpy.array(self.fold_train_indices_mask)


    def get_kfold_indices(self):
        """
        Returns the indices for rating matrix for each kfold split. Where each test set
        contains ~1/k of the total items a user has in their digital library.

        :returns: a list of all indices of the training set and test set.
        :rtype: list of lists
        """
        if self.random_seed is False:
            numpy.random.seed(42)

        test_indices = []

        for user in range(self.ratings.shape[0]):

            # Indices for all items in the rating matrix.
            item_indices = numpy.arange(self.ratings.shape[1])

            # Indices of all items in user's digital library.
            rated_items_indices = self.ratings[user].nonzero()[0]
            mask = numpy.ones(len(self.ratings[user]), dtype=bool)
            mask[[rated_items_indices]] = False
            # Indices of all items not in user's digital library.
            non_rated_indices = item_indices[mask]

            # Shuffle all rated items indices
            numpy.random.shuffle(rated_items_indices)

            # Size of 1/k of the total user's ratings
            size_of_test = round((1.0 / self.k_folds) * len(rated_items_indices))

            # 2d List that stores all the indices of each test set for each fold.
            test_ratings = [[] for x in range(self.k_folds)]

            counter = 0
            numpy.random.shuffle(non_rated_indices)
            # List that stores the number of indices to be added to each test set.
            num_to_add = []

            # create k different folds for each user.
            for index in range(self.k_folds):
                if index == self.k_folds - 1:
                    test_ratings[index] = numpy.array(rated_items_indices[counter:len(rated_items_indices)])
                else:
                    test_ratings[index] = numpy.array(rated_items_indices[counter:counter + size_of_test])
                counter += size_of_test

                # adding unique zero ratings to each test set
                num_to_add.append(int((self.ratings.shape[1] / self.k_folds) - len(test_ratings[index])))
                if index > 0 and num_to_add[index] != num_to_add[index - 1]:
                    addition = non_rated_indices[index * (num_to_add[index - 1]):
                                                         (num_to_add[index - 1] * index) + num_to_add[index]]
                else:
                    addition = non_rated_indices[index * (num_to_add[index]):num_to_add[index] * (index + 1)]

                test_ratings[index] = numpy.append(test_ratings[index], addition)
                test_indices.append(test_ratings[index])

        self.test_indices = test_indices
        return test_indices

    def generate_kfold_matrix(self, test_indices):
        """
        Returns a training set and a training set matrix for one fold.
        This method is to be used in conjunction with get_kfold_indices()

        :param int[] test_indices: array of test set indices.
        :returns: Training set matrix and Test set matrix.
        :rtype: 2-tuple of 2d numpy arrays
        """
        train_matrix = numpy.zeros(self.ratings.shape)
        test_matrix = numpy.zeros(self.ratings.shape)

        # Save the mask of train indices
        self.fold_train_indices_mask = []

        for user in range(train_matrix.shape[0]):
            train_indices = list(set(range(self.n_items)) - set(test_indices[user]))
            test_matrix[user, test_indices[user]] = self.ratings[user, test_indices[user]]
            train_matrix[user, train_indices] = self.ratings[user, train_indices]

            # Create the mask of training data
            train_indices_mask = numpy.zeros(len(self.ratings[user]), dtype=bool)
            train_indices_mask[train_indices] = True
            self.fold_train_indices_mask.append(train_indices_mask)

        return train_matrix, test_matrix

    def load_top_recommendations(self, n_recommendations, predictions, test_data, fold):
        """
        This method loads the top n recommendations into a local variable.

        :param int n_recommendations: number of recommendations to be generated.
        :param int[][] predictions: predictions matrix (only 0s or 1s)
        :returns: A matrix of top recommendations for each user.
        :rtype: int[][]
        """
        for user in range(self.ratings.shape[0]):
            # Bug discovered by Mostafa
            #nonzeros = self.test_indices[(user * (1 + fold))]
            #Fix:
            nonzeros = self.test_indices[user * self.k_folds + fold]
            top_recommendations = TopRecommendations(n_recommendations)
            for index in nonzeros:
                index = int(index)
                top_recommendations.insert(index, predictions[user][index])
            self.recommendation_indices[user] = list(reversed(top_recommendations.get_indices()))
            top_recommendations = None

        self.recs_loaded = True
        return self.recommendation_indices

    def get_rmse(self, predicted, actual=None):
        """
        The method given a prediction matrix returns the root mean squared error (rmse).

        :param float[][] predicted: numpy matrix of floats representing the predicted ratings
        :returns: root mean square error
        :rtype: float
        """
        if actual is None:
            actual = self.ratings

        rss = 0
        for i in range(predicted.shape[0]):
            rss += numpy.sum((predicted[i] - actual[i]) ** 2)
        rss = float(rss) / numpy.size(predicted)

        return numpy.sqrt(rss)

    def calculate_recall(self, ratings, predictions):
        """
        The method given original ratings and predictions returns the recall of the recommender

        :param int[][] ratings: ratings matrix
        :param int[][] predictions: predictions matrix (only 0s or 1s)
        :returns: recall, ranges from 0 to 1
        :rtype: float
        """
        denom = sum(sum(ratings))
        nonzeros = ratings.nonzero()
        nonzeros_predictions = predictions[nonzeros]
        return sum(nonzeros_predictions) / denom  # Division by zeros are handled.

    def recall_at_x(self, x, predictions, ratings, rounded_predictions):
        """
        The method calculates the average recall of all users by only looking at the top x
        and the normalized Discounted Cumulative Gain.

        :param int x: number of recommendations to look at, sorted by relevance.
        :param int[][] ratings: ratings matrix
        :param float[][] predictions: calculated predictions of the recommender.
        :param int[][] test_data: test data.
        :returns: Recall at x
        :rtype: float
        """

        recalls = []
        for user in range(ratings.shape[0]):
            recommendation_hits = 0
            user_likes = ratings[user].sum()
            recall = 0
            if user_likes != 0:
                recommendation_hits = (self.ratings[user][self.recommendation_indices[user][:x]] *
                                       rounded_predictions[user][self.recommendation_indices[user][:x]]).sum()

                recall = recommendation_hits / (min(x, user_likes) * 1.0)
                recalls.append(recall)
        return numpy.mean(recalls, dtype=numpy.float16)

    def recall_ndcg_mrr_at_list_of_x(self, recall_list, ndcg_list, mrr_list, ratings, rounded_predictions):
        """
        The method calculates the average recall@x of all users, for a given list of x values

        :param int recall_list: the list of numbers of recommendations to look at, sorted by relevance.
        :param int[][] ratings: ratings matrix
        :param float[][] rounded_predictions: rounded predictions (zeros or ones) of the recommender.

        :returns: list of Recall at x
        :rtype: numpy array of float
        """

        max_num_recommendations = max(recall_list + mrr_list + ndcg_list)
        users_recalls = []
        users_mrrs=[]
        users_ndcgs=[]
        for user in range(ratings.shape[0]):
            user_recalls = []
            user_mrrs = []
            user_ndcgs = []
            recommendation_hits = 0
            user_likes = ratings[user].sum()
            recommended_ids = self.recommendation_indices[user]

            mrr_score = 0
            dcg_score = 0
            idcg = 0
            for i, rec_index in enumerate (range(min(len(recommended_ids),max_num_recommendations))):

                # recall
                if (i+1) in recall_list:
                    recall = 0
                    if user_likes != 0:
                        recommendation_hits = (self.ratings[user][recommended_ids[:(i+1)]] *
                                               rounded_predictions[user][recommended_ids[:(i+1)]]).sum()

                        recall = recommendation_hits / (min((i+1), user_likes) * 1.0)
                        user_recalls.append(recall)

                # MRR:
                if (i+1) <= max(mrr_list):
                    if mrr_score == 0:
                        mrr_score = self.ratings[user][rec_index] * rounded_predictions[user][rec_index] / (i+1)
                    if (i+1) in mrr_list:
                        user_mrrs.append(mrr_score)

                # ndcg:
                if (i+1) <= max(ndcg_list):
                    dcg_score += (self.ratings[user, rec_index] * rounded_predictions[user][rec_index]) / numpy.log2(i + 2)
                    idcg += 1 / numpy.log2(i + 2)
                    if (i +1) in ndcg_list:
                        user_ndcgs.append(dcg_score/idcg)

            users_recalls.append(user_recalls)
            users_ndcgs.append(user_ndcgs)
            users_mrrs.append(user_mrrs)

        #print(numpy.mean(users_recalls, axis = 0))
        #print(numpy.mean(users_ndcgs, axis=0))
        #print(numpy.mean(users_mrrs, axis =0))

        return ( numpy.mean(users_recalls, axis = 0), numpy.mean(users_ndcgs, axis=0), numpy.mean(users_mrrs, axis =0) )

    def calculate_ndcg(self, n_recommendations, predictions, test_data, rounded_predictions):
        """
        The method calculates the normalized Discounted Cumulative Gain of all users
        by only looking at the top n_recommendations.

        :param int n_recommendations: number of recommendations to look at, sorted by relevance.
        :param float[][] predictions: calculated predictions of the recommender
        :returns: nDCG for n_recommendations
        :rtype: float
        """
        ndcgs = []
        for user in range(self.ratings.shape[0]):
            dcg = 0
            idcg = 0
            for pos_index, index in enumerate( self.recommendation_indices[user] ):
                dcg += (self.ratings[user, index] * rounded_predictions[user][index]) / numpy.log2(pos_index + 2)
                idcg += 1 / numpy.log2(pos_index + 2)
                if pos_index + 1 == n_recommendations:
                    break
            if idcg != 0:
                ndcgs.append(dcg / idcg)
        return numpy.mean(ndcgs, dtype=numpy.float16)

    def calculate_mrr(self, n_recommendations, predictions, test_data, rounded_predictions):
        """
        The method calculates the mean reciprocal rank for all users
        by only looking at the top n_recommendations.

        :param int n_recommendations: number of recommendations to look at, sorted by relevance.
        :param float[][] predictions: calculated predictions of the recommender
        :returns: mrr at n_recommendations
        :rtype: float
        """

        mrr_list = []

        for user in range(self.ratings.shape[0]):
            mrr = 0
            for mrr_index, index in enumerate(self.recommendation_indices[user]):
                score = self.ratings[user][index] * rounded_predictions[user][index]
                if score == 1:
                    mrr = score / (mrr_index + 1)
                    break
                if mrr_index + 1 == n_recommendations:
                    break
            mrr_list.append(mrr)

        return numpy.mean(mrr_list, dtype=numpy.float16)
