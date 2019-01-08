import numpy as np
from typing import List
from classifier import Classifier
import sys


class DecisionTree(Classifier):
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features: List[List[float]], labels: List[int]):
        # init.
        assert (len(features) > 0)
        self.feautre_dim = len(features[0])
        num_cls = np.max(labels) + 1

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features: List[List[float]]) -> List[int]:
        y_pred = []
        for feature in features:
            y_pred.append(self.root_node.predict(feature))
        return y_pred

    def print_tree(self, node=None, name='node 0', indent=''):
        if node is None:
            node = self.root_node
        print(name + '{')

        string = ''
        for idx_cls in range(node.num_cls):
            string += str(node.labels.count(idx_cls)) + ' '
        print(indent + ' num of sample / cls: ' + string)

        if node.splittable:
            print(indent + '  split by dim {:d}'.format(node.dim_split))
            for idx_child, child in enumerate(node.children):
                self.print_tree(node=child, name='  ' + name + '/' + str(idx_child), indent=indent + '  ')
        else:
            print(indent + '  cls', node.cls_max)
        print(indent + '}')


class TreeNode(object):
    def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls

        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label  # majority of current node

        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    def split(self):

        # calculates  the conditional entropy
        def conditional_entropy(branches: List[List[int]]) -> float:

            feature_data = np.array(branches)
            feature_data_sum_vertical = np.sum(feature_data, axis=0)
            total_sum = np.sum(feature_data_sum_vertical)
            feature_data = np.divide(feature_data, feature_data_sum_vertical)
            feature_data = feature_data * np.log2(feature_data, where=feature_data != 0)
            feature_data = -np.sum(feature_data, axis=0)
            feature_data = np.dot(feature_data, feature_data_sum_vertical)
            c_entropy = np.sum(feature_data) / total_sum
            return float(c_entropy)

        # creates a C x B matrix to send it to entropy function to calculate the entropy
        def getBranches(features, labels):
            feature_unique = np.unique(features)
            feature_unique_len = np.unique(features).size
            label_unique = np.unique(labels)
            label_unique_len = np.unique(labels).size
            feature_unique_dict = dict()
            label_unique_dict = dict()
            i = 0
            for label in label_unique:
                label_unique_dict[label] = i
                i += 1
            i = 0
            for feature in feature_unique:
                feature_unique_dict[feature] = i
                i += 1

            branches = np.zeros((label_unique_len, feature_unique_len))
            for index in range(0, len(features)):
                branches[label_unique_dict[labels[index]]][feature_unique_dict[features[index]]] += 1
            return branches, feature_unique

        # stores the min entropy and its feature index
        min_entropy = sys.maxsize
        min_feature_unique = list()
        min_index = None

        for idx_dim in range(len(self.features[0])):

            current_feature_data = np.array(self.features)
            current_feature_data = current_feature_data[:, idx_dim]
            branches, feature_unique = getBranches(current_feature_data, self.labels)
            if len(feature_unique) < 2:
                continue
            cond_entropy = conditional_entropy(list(branches))

            if cond_entropy < min_entropy:
                min_entropy = cond_entropy
                min_index = idx_dim
                min_feature_unique = list(feature_unique)

        if min_index is None:
            self.splittable = False
            return

        self.dim_split = min_index
        self.feature_uniq_split = list(min_feature_unique)

        full_data = np.c_[self.features, self.labels]

        for value in self.feature_uniq_split:
            child_data = full_data[full_data[:, self.dim_split] == value]
            child_label = child_data[:, -1]
            child_feature = child_data[:, :-1]
            # new added line
            child_feature = np.delete(child_feature, self.dim_split, axis=1)
            num_cls = np.unique(child_label).size
            child = TreeNode(list(child_feature), list(child_label), self.num_cls)
            self.children.append(child)

        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()

        return

    def predict(self, feature: List[int]) -> int:
        if self.splittable:
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            feature = feature[:self.dim_split] + feature[self.dim_split + 1:]
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max
