from typing import Optional

import numpy as np
from dataclasses import dataclass

@dataclass
class Node:
    threshold: Optional[float] = None
    feature_index: Optional[int] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    value: float = None

@dataclass
class Split:
    left: []
    right: []
    G_L :[]
    H_L : []
    G_R:[]
    H_R:[]
    threshold: Optional[float] = None
    feature_index: Optional[int] = None


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_hessian(p):
    return p*(1-p)

def compute_weight(G, H, regularization):
    return -(np.sum(G)/(np.sum(H)+regularization))



class ClassificationTree:

    def __init__(self, max_depth=3, min_child_weight=0.4, regularization=1, split_gain=0.1):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.regularization = regularization
        self.split_gain = split_gain
        self.node: Optional[Node] = None

    def fit(self, X, G, H):
        self.node = self.__build_tree(X=X, G=G, H=H, depth=1)
        return self


    def __find_best_split(self,X, G, H):
        row_count, feature_count = X.shape
        best_split = None
        best_gain = 0
        G_P = np.sum(G)
        H_P = np.sum(H)
        for feature_index in range(feature_count):
            sorted_value = np.unique(X[:, feature_index])
            feature_value = X[:, feature_index]
            thresholds = (sorted_value[:-1]+sorted_value[1:])/2
            for threshold in thresholds:
                left_mask = (feature_value<threshold)
                right_mask = (feature_value>=threshold)
                left_split = np.where(feature_value<threshold)[0]
                right_split =  np.where(feature_value>=threshold)[0]
                G_L = np.sum(G[left_mask])
                H_L= np.sum(H[left_mask])
                G_R = np.sum(G[right_mask])
                H_R = np.sum(H[right_mask])
                if H_L<self.min_child_weight or H_R<self.min_child_weight:
                    continue
                gain= 1 / 2 * ((G_L**2/(H_L+self.regularization)) + (G_R**2/(H_R+self.regularization))-(G_P**2/(H_P+self.regularization)))- self.split_gain
                if gain > best_gain:
                   best_gain = gain
                   best_split = Split(left=left_split, right=right_split, threshold=threshold, feature_index=feature_index, G_L=G[left_mask], H_L=H[left_mask], G_R=G[right_mask], H_R=H[right_mask])
        return best_split

    def __build_tree(self, X,G, H, depth):

        if depth>self.max_depth:
            return Node(value=compute_weight(G, H, self.regularization))

        if np.sum(H)<self.min_child_weight:
            return Node(value=compute_weight(G, H, self.regularization))

        split = self.__find_best_split(X=X, G=G, H=H)

        if split is None:
            return Node(value=compute_weight(G, H, self.regularization))

        left_node = self.__build_tree(X=X[split.left], G= split.G_L, H=split.H_L, depth=depth+1)
        right_node = self.__build_tree(X=X[split.right], G= split.G_R, H=split.H_R, depth=depth+1)
        return Node(threshold=split.threshold, feature_index=split.feature_index, left=left_node, right=right_node)

    def _predict(self, X):
        predictions = []
        for x in X:
            current = self.node
            while current.value is None:
                if x[current.feature_index]<current.threshold:
                    current = current.left
                else:
                    current = current.right
            predictions.append(current.value)
        return predictions


def compute_gradient(p, y):
    return p-y

class XgBoostRecursion:

    def __init__(self, n_estimator=5, max_depth=3, lr=0.1, min_child_weight=0.4, regularization=1, split_gain=0.2):
        self.n_estimator=n_estimator
        self.max_depth=max_depth
        self.lr=lr
        self.min_child_weight = min_child_weight
        self.regularization = regularization
        self.split_gain = split_gain
        self.tree = []
        self.initial_z = []
        self.p = None
        self.current_z = None

    def compute_initial_z(self, Y):
        self.initial_z=np.full(len(Y), 0, dtype=float)

    def _fit(self, X, Y):
        self.compute_initial_z(Y=Y)
        z = self.initial_z
        for _ in range(self.n_estimator):
            p = sigmoid(np.array(z))
            G = compute_gradient(p, Y)
            H = compute_hessian(p)
            classification_tree = ClassificationTree(max_depth=self.max_depth, min_child_weight=self.min_child_weight, regularization=self.regularization, split_gain=self.split_gain)
            tree = classification_tree.fit(X, G, H)
            z = z + self.lr*np.asarray(tree._predict(X=X))
            self.current_z=z
            self.p = sigmoid(np.array(z))
            print(tree.node)
            print(self.p)
            self.tree.append(tree)




X = np.array([
    [22, 50, 620],
    [25, 80, 580],
    [28, 40, 700],
    [30, 90, 650],
    [35, 30, 640],
    [40, 70, 610],
    [45, 45, 730],
    [50, 60, 590]
], dtype=float)

y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
], dtype=float)

gradient_boosting = XgBoostRecursion(5, 3, 0.1)
gradient_boosting._fit(X, y)






