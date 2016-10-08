from sklearn import ensemble
import xgboost as xgb

from .optimizer.optimizer import HyperoptOptimizer
from .optimizer.task import Task
from .spaces import xgboost, random_forest


class XGBoostOptimizer(HyperoptOptimizer):
    def __init__(self, task: Task, scorer):
        if task.task == "classification":
            space = xgboost.classification_space()
        else:
            space = xgboost.general_space()
        super().__init__(xgb.XGBModel(), task, space, scorer)

    def get_prediction(self, model, X):
        return model.predict(X)

    def process_parameters(self, parameters):
        parameters["max_depth"] = int(parameters["max_depth"])
        parameters["n_estimators"] = int(parameters["n_estimators"])
        return parameters


class RandomForestOptimizer(HyperoptOptimizer):
    def __init__(self, task: Task, scorer):
        if task.task == "classification":
            space = random_forest.classification_space()
            model = ensemble.RandomForestClassifier()
        else:
            space = random_forest.regression_space()
            model = ensemble.RandomForestRegressor()
        super().__init__(model, task, space, scorer)

    def get_prediction(self, model, X):
        if self.task.task == "classification":
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X)

    def process_parameters(self, parameters):
        if parameters["max_depth"] is not None:
            parameters["max_depth"] = int(parameters["max_depth"])
        parameters["n_estimators"] = int(parameters["n_estimators"])
        return parameters



