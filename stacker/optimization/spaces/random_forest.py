from stacker.optimization.spaces.space import Space
from stacker.optimization.parameters.random_forest import *
from stacker.optimization.parameters.parameter_distribution import *

import numpy as np


def classification_space():
    space = {
        n_estimators: QUniform(n_estimators, 10, 2000, 1),
        criterion: Choice(criterion, ["gini", "entropy"]),
        max_features: Choice(max_features, [None, "log2", "sqrt", "auto"] + list(np.arange(0.05, 1., 0.05))),
        max_depth: Choice(max_depth, [None] + list(range(5, 50))),
        min_samples_split: QUniform(min_samples_split, 1, 10, 1),
        min_samples_leaf: QUniform(min_samples_leaf, 1, 10, 1),
        min_weight_fraction_leaf: Uniform(min_weight_fraction_leaf, 0.0, 0.1),
        max_leaf_nodes: Choice(max_leaf_nodes, [None] + list(range(1, 1001))),
        bootstrap: Choice(bootstrap, [False, True]),
        class_weight: Choice(class_weight, [None, "balanced"])
    }
    return Space(space)


def regression_space():
    space = {
        n_estimators: QUniform(n_estimators, 10, 2000, 1),
        max_features: Choice(max_features, [None, "log2", "sqrt", "auto"] + list(np.arange(0.05, 1., 0.05))),
        max_depth: Choice(max_depth, [None] + list(range(5, 50))),
        min_samples_split: QUniform(min_samples_split, 1, 10, 1),
        min_samples_leaf: QUniform(min_samples_leaf, 1, 10, 1),
        min_weight_fraction_leaf: Uniform(min_weight_fraction_leaf, 0.0, 0.1),
        max_leaf_nodes: Choice(max_leaf_nodes, [None] + list(range(1, 1001))),
        bootstrap: Choice(bootstrap, [False, True]),
    }
    return Space(space)