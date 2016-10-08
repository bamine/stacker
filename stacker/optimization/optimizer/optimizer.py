import numpy as np
import xgboost as xgb
import logging
from sklearn.base import BaseEstimator
from sklearn import ensemble
from hyperopt import STATUS_OK, fmin, tpe, STATUS_FAIL, space_eval

from ..optimizer.task import Task
from ..spaces import xgboost, random_forest
from ..spaces.space import Space

logger = logging.getLogger('optimization')


class Optimizer:
    def __init__(self, model: BaseEstimator, task: Task, space: Space, scorer):
        self.task = task
        self.scorer = scorer
        self.space = space
        self.model = model
        self.best = None

    def get_prediction(self, model, X):
        return model.predict(X)

    def process_parameters(self, parameters):
        return parameters

    def score(self, parameters):
        parameters = self.process_parameters(parameters)
        if self.task.validation_method == "train_test_split":
            return self.score_train_test_split(parameters)
        elif self.task.validation_method == "cv":
            return self.score_cv(parameters)

    def start_optimization(self, max_evals):
        return NotImplemented

    def score_train_test_split(self, parameters):
        logger.info("Evaluating with test size %s with parameters %s", self.task.test_size, parameters)
        logger.info("Training model ...")
        self.model.set_params(**parameters)
        self.model.fit(self.task.X_train, self.task.y_train)
        logger.info("Training model done !")
        y_pred = self.get_prediction(self.model, self.task.X_test)
        score = self.scorer(self.task.y_test, y_pred)
        logger.info("Score = %s", score)
        return {'loss': score, 'status': STATUS_OK}

    def score_cv(self, parameters):
        logger.info("Evaluating using %s-fold CV with parameters %s", self.task.kfold.n_folds, parameters)
        self.model.set_params(**parameters)
        scores = []
        for i, (train_index, test_index) in enumerate(self.task.kfold):
            logger.info("Starting fold %s ...", i)
            X_train, X_test = self.task.X[train_index], self.task.X[test_index]
            y_train, y_test = self.task.y[train_index], self.task.y[test_index]
            self.model.fit(X_train, y_train)
            logger.info("Training for fold %s done !", i)
            y_pred = self.get_prediction(self.model, X_test)
            score = self.scorer(y_test, y_pred)
            logger.info("Score %s", score)
            scores.append(score)
        logger.info("Cross validation done !")
        mean_score = np.mean(scores)
        logger.info("Mean Score = %s", mean_score)
        return {'loss': mean_score, 'status': STATUS_OK}


class HyperoptOptimizer(Optimizer):
    def start_optimization(self, max_evals):
        logger.info("Started optimization for task %s", self.task)
        space = self.space.hyperopt()
        best = fmin(self.score, space, algo=tpe.suggest, max_evals=max_evals)
        self.best = space_eval(space, best)
        return self.best