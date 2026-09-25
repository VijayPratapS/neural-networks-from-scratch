import  numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Node:
    feature_index : Optional[int]=None
    threshold : Optional[float]=None
    value : Optional[float]=None
    left : Optional['Node']=None
    right : Optional['Node']=None

@dataclass
class Split:
    feature_index: int
    threshold: float
    score: float


class RegressionTree:
    def __init__(self, max_depth=1, min_samples_split=2, min_samples_leaf=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X,Y):
        self.root= self._build_tree(X, Y, depth=0)
        return self

    def calculate_split_score(self, y):
        prediction = np.mean(y)
        return np.sum((y-prediction)**2)

    def _build_tree(self, x, y, depth):
        n_samples = len(y)
        if depth>=self.max_depth:
            return Node(value=np.mean(y))

        if n_samples < self.min_samples_split:
            return Node(value=np.mean(y))

        if np.all(y==y[0]):
            return Node(value=np.mean(y))

        best_split = self._find_best_split(x, y)
        print(
            f"Depth={depth}, "
            f"Feature={best_split.feature_index}, "
            f"Threshold={best_split.threshold:.2f}, "
            f"SSE={best_split.score:.2f}"
        )
        if best_split is None:
            return Node(value= np.mean(y))
        feature_value = x[:, best_split.feature_index]

        x_mask = (feature_value<best_split.threshold)
        x_left = x[x_mask]
        y_left = y[x_mask]

        y_mask = (feature_value >= best_split.threshold)
        x_right = x[y_mask]
        y_right = y[y_mask]

        left_child = self._build_tree(x_left, y_left, depth+1)
        right_child = self._build_tree(x_right, y_right, depth+1)

        return Node(feature_index=best_split.feature_index, threshold=best_split.threshold, left=left_child, right=right_child)

    def _find_best_split(self, x, y):
        row_index, feature_index= x.shape
        best_score = np.inf
        split = None
        for index in range(feature_index):
            column_data = x[:, index]
            unique_values = np.unique(column_data)

            thresholds = (
                                 unique_values[:-1]
                                 +
                                 unique_values[1:]
                         ) / 2
            for threshold in thresholds:
                left = (column_data<threshold)
                right = (column_data>=threshold)
                y_left = y[left]
                y_right = y[right]
                if len(y_left)<self.min_samples_leaf:
                    continue
                if len(y_right)<self.min_samples_leaf:
                    continue
                y_left_score = self.calculate_split_score(y_left)
                y_right_score = self.calculate_split_score(y_right)
                total_score = y_left_score+y_right_score
                if total_score<best_score:
                    best_score = total_score
                    split=Split(feature_index=index, score=total_score, threshold=threshold)

        return split

    def __predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] < node.threshold:
           return self.__predict_one(x, node.left)
        else:
            return self.__predict_one(x, node.right)

    def predict(self, X):
        node = self.root
        predictions=[]
        for x in X:
            predictions.append(self.__predict_one(x, node))
        return np.array(predictions)

def initial_prediction(Y):
    y_arrays = np.asarray(Y, dtype=float)
    return np.full(len(Y),np.mean(y_arrays), float)

def calculate_residuals(actual, predicted):
    return actual-predicted


class GradientBoostingRegressorFromScratch:
    def __init__(self,
                 n_estimator=10, learning_rate=0.1, max_depth=2, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimator
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = (
            min_samples_split
        )
        self.min_samples_leaf = (
            min_samples_leaf
        )
        self.initial_prediction = None
        self.trees = []
        self.training_losses = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y=np.asarray(y, dtype=float)
        self.initial_prediction = initial_prediction(y)
        for tree_number in range(self.n_estimators):
            residuals = calculate_residuals(y, self.initial_prediction)
            tree = RegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X, residuals)
            coorection = tree.predict(X)
            self.initial_prediction = self.initial_prediction+(self.learning_rate*coorection)
            self.trees.append(tree)
            errors = y-self.initial_prediction
            mse = np.mean(errors**2)
            self.training_losses.append(mse)
            print(f'\ntree number{tree_number+1}')
            print(
                "Residuals:",
                residuals
            )
            print(
                "Tree corrections:",
                coorection
            )

            print(
                "New predictions:",
                self.initial_prediction
            )

        return self




X = np.array([
    [22, 30, 600],
    [25, 35, 620],
    [28, 80, 700],
    [35, 40, 650],
    [40, 90, 760],
    [45, 95, 780],
    [50, 45, 680],
    [55, 100, 800]
], dtype=float)

y = np.array([
    20,
    25,
    70,
    35,
    90,
    100,
    45,
    110
], dtype=float)

gradient_boosting = GradientBoostingRegressorFromScratch(5, 0.5, 2)
gradient_boosting.fit(X, y)





