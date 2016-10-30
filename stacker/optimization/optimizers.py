from sklearn import ensemble
import xgboost as xgb
from copy import deepcopy
import numpy as np

from stacker.optimization.parameters.parameter import Parameter
from .optimizer.optimizer import HyperoptOptimizer
from .optimizer.task import Task
from .optimizer.scorer import Scorer
from .spaces.space import Space
from .parameters.parameter_distribution import *
from .logging.optimization_logger import OptimizationLogger
from .loggers import VoidLogger


class XGBoostOptimizer(HyperoptOptimizer):

    class Params:
        max_depth = Parameter("max_depth")
        learning_rate = Parameter("learning_rate")
        n_estimators = Parameter("n_estimators")
        objective = Parameter("objective")
        gamma = Parameter("gamma")
        min_child_weight = Parameter("min_child_weight")
        max_delta_step = Parameter("max_delta_step")
        subsample = Parameter("subsample")
        colsample_bytree = Parameter("colsample_bytree")
        colsample_bylevel = Parameter("colsample_bylevel")
        reg_alpha = Parameter("reg_alpha")
        reg_lambda = Parameter("reg_lambda")
        scale_pos_weight = Parameter("scale_pos_weight")
        base_score = Parameter("base_score")

        general_space = Space({
                max_depth: Uniform(max_depth, 3, 50),
                learning_rate: QUniform(learning_rate, 0.0001, 0.5, 0.0001),
                n_estimators: Uniform(n_estimators, 10, 5000),
                gamma: Uniform(gamma, 0, 10),
                min_child_weight: Uniform(min_child_weight, 0.1, 10),
                max_delta_step: Uniform(max_delta_step, 0, 10),
                subsample: Uniform(subsample, 0.1, 1),
                colsample_bytree: Uniform(colsample_bytree, 0.1, 1),
                colsample_bylevel: Uniform(colsample_bylevel, 0.5, 1),
                reg_alpha: LogUniform(reg_alpha, -5, 5),
                reg_lambda: LogUniform(reg_lambda, -5, 5),
                scale_pos_weight: Uniform(scale_pos_weight, 0, 100),
                base_score: Uniform(base_score, 0, 1),
            })

        classification_space = deepcopy(general_space)
        classification_space.fix_parameters([reg_alpha, reg_lambda])

    def __init__(self, task: Task, scorer: Scorer, opt_logger: OptimizationLogger=VoidLogger(None)):
        if task.task == "classification":
            space = XGBoostOptimizer.Params.classification_space
        else:
            space = XGBoostOptimizer.Params.general_space
        super().__init__(xgb.XGBModel(), task, space, scorer, opt_logger)

    def get_prediction(self, model, X):
        return model.predict(X)

    def process_parameters(self, parameters):
        parameters["max_depth"] = int(parameters["max_depth"])
        parameters["n_estimators"] = int(parameters["n_estimators"])
        return parameters


class RandomForestOptimizer(HyperoptOptimizer):

    class Params:
        n_estimators = Parameter("n_estimators")
        criterion = Parameter("criterion")
        max_features = Parameter("max_features")
        max_depth = Parameter("max_depth")
        min_samples_split = Parameter("min_samples_split")
        min_samples_leaf = Parameter("min_samples_leaf")
        min_weight_fraction_leaf = Parameter("min_weight_fraction_leaf")
        max_leaf_nodes = Parameter("max_leaf_nodes")
        bootstrap = Parameter("bootstrap")
        class_weight = Parameter("class_weight")

        classification_space = Space({
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
            })

        regression_space = Space({
                n_estimators: QUniform(n_estimators, 10, 2000, 1),
                max_features: Choice(max_features, [None, "log2", "sqrt", "auto"] + list(np.arange(0.05, 1., 0.05))),
                max_depth: Choice(max_depth, [None] + list(range(5, 50))),
                min_samples_split: QUniform(min_samples_split, 1, 10, 1),
                min_samples_leaf: QUniform(min_samples_leaf, 1, 10, 1),
                min_weight_fraction_leaf: Uniform(min_weight_fraction_leaf, 0.0, 0.1),
                max_leaf_nodes: Choice(max_leaf_nodes, [None] + list(range(1, 1001))),
                bootstrap: Choice(bootstrap, [False, True]),
            })

    def __init__(self, task: Task, scorer: Scorer, opt_logger: OptimizationLogger=VoidLogger(None)):
        if task.task == "classification":
            space = RandomForestOptimizer.Params.classification_space
            model = ensemble.RandomForestClassifier()
        else:
            space = RandomForestOptimizer.Params.regression_space
            model = ensemble.RandomForestRegressor()
        super().__init__(model, task, space, scorer, opt_logger)

    def get_prediction(self, model, X):
        if self.task.task == "classification":
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X)

    def process_parameters(self, parameters):
        if parameters["max_depth"] is not None:
            parameters["max_depth"] = int(parameters["max_depth"])
        parameters["n_estimators"] = int(parameters["n_estimators"])
        parameters["min_samples_split"] = int(parameters["min_samples_split"])
        parameters["min_samples_leaf"] = int(parameters["min_samples_leaf"])
        return parameters



