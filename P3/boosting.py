import numpy as np
from typing import List, Set
from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod
import sys
import math


class Boosting(Classifier):
    # Boosting from pre-defined classifiers
    def __init__(self, clfs: Set[Classifier], T=0):
        self.clfs = clfs  # set of weak classifiers to be considered
        self.num_clf = len(clfs)
        if T < 1:
            self.T = self.num_clf
        else:
            self.T = T

        self.clfs_picked = []  # list of classifiers h_t for t=0,...,T-1
        self.betas = []  # list of weights beta_t for t=0,...,T-1
        return

    @abstractmethod
    def train(self, features: List[List[float]], labels: List[int]):
        return

    def predict(self, features: List[List[float]]) -> List[int]:
        '''
        Inputs:
        - features: the features of all test examples

        Returns:
        - the prediction (-1 or +1) for each example (in a list)
        '''

        predictions = np.zeros(len(features))
        for t in range(0, self.T):
            predictions += self.betas[t] * np.array(self.clfs_picked[t].predict(features))
        predictions = np.sign(predictions)
        predictions = list(predictions)
        return predictions

    ########################################################
    # TODO: implement "predict"
    ########################################################


class AdaBoost(Boosting):
    def __init__(self, clfs: Set[Classifier], T=0):
        Boosting.__init__(self, clfs, T)
        self.clf_name = "AdaBoost"
        return

    def train(self, features: List[List[float]], labels: List[int]):
        '''
        Inputs:
        - features: the features of all examples
        - labels: the label of all examples

        Require:
        - store what you learn in self.clfs_picked and self.betas
        '''

        feature_len = len(features)
        d_weight = [1 / feature_len for i in range(feature_len)]

        for t in range(0, self.T):

            h_min_at_t = sys.maxsize
            clf_min_at_t = None
            predictions_min_at_t = list()
            for clf in self.clfs:
                predictions = clf.predict(features)
                predictions_sum = 0
                for i in range(0, len(predictions)):
                    if predictions[i] != labels[i]:
                        predictions_sum += d_weight[i]
                if predictions_sum < h_min_at_t:
                    h_min_at_t = predictions_sum
                    clf_min_at_t = clf
                    predictions_min_at_t = predictions

            error_at_t = h_min_at_t
            beta_at_t = (1 / 2) * (np.log((1 - error_at_t) / error_at_t))

            for i in range(feature_len):
                if predictions_min_at_t[i] == labels[i]:
                    d_weight[i] = d_weight[i] * math.exp(-beta_at_t)
                else:
                    d_weight[i] = d_weight[i] * math.exp(beta_at_t)

            d_weight_at_t_sum = sum(d_weight)
            for i in range(0, len(d_weight)):
                d_weight[i] = d_weight[i] / d_weight_at_t_sum

            self.clfs_picked.append(clf_min_at_t)
            self.betas.append(beta_at_t)

    ############################################################
    # TODO: implement "train"
    ############################################################

    def predict(self, features: List[List[float]]) -> List[int]:
        return Boosting.predict(self, features)
