import numpy as np
import logging
from sklearn.base import BaseEstimator
from hyperopt import STATUS_OK, fmin, tpe, space_eval

from ..optimizer.task import Task
from ..optimizer.result import OptimizationResult
from ..optimizer.scorer import Scorer
from ..logging.optimization_logger import OptimizationLogger
from ..spaces.space import Space

logger = logging.getLogger('optimization')


class Optimizer:
    def __init__(self, model: BaseEstimator, task: Task, space: Space, scorer: Scorer, opt_logger: OptimizationLogger):
        self.model = model
        self.task = task
        self.space = space
        self.scorer = scorer
        self.opt_logger = opt_logger
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
        score = self.scorer.scoring_function(self.task.y_test, y_pred)
        logger.info("Score = %s", score)
        result = OptimizationResult(
            task=self.task.name,
            model=str(self.model),
            parameters=parameters,
            score=score,
            scorer_name=self.scorer.name,
            validation_method=self.task.validation_method,
            predictions=y_pred.tolist(),
            random_state=self.task.random_state)
        self.opt_logger.save(result)
        return {'loss': score, 'status': STATUS_OK}

    def score_cv(self, parameters):
        logger.info("Evaluating using %s-fold CV with parameters %s", self.task.kfold.n_folds, parameters)
        self.model.set_params(**parameters)
        scores = []
        fold_predictions = []
        for i, (train_index, test_index) in enumerate(self.task.kfold):
            logger.info("Starting fold %s ...", i)
            X_train, X_test = self.task.X[train_index], self.task.X[test_index]
            y_train, y_test = self.task.y[train_index], self.task.y[test_index]
            self.model.fit(X_train, y_train)
            logger.info("Training for fold %s done !", i)
            y_pred = self.get_prediction(self.model, X_test)
            fold_predictions.append(y_pred.tolist())
            score = self.scorer.scoring_function(y_test, y_pred)
            logger.info("Score %s", score)
            scores.append(score)
        logger.info("Cross validation done !")
        mean_score = np.mean(scores)
        logger.info("Mean Score = %s", mean_score)
        result = OptimizationResult(
            model=str(self.model),
            parameters=parameters,
            score=mean_score,
            scorer_name=self.scorer.name,
            validation_method=self.task.validation_method,
            predictions=fold_predictions,
            random_state=self.task.random_state)
        self.opt_logger.save(result)
        return {'loss': mean_score, 'status': STATUS_OK}


class HyperoptOptimizer(Optimizer):
    def start_optimization(self, max_evals):
        logger.info("Started optimization for task %s", self.task)
        space = self.space.hyperopt()
        best = fmin(self.score, space, algo=tpe.suggest, max_evals=max_evals)
        self.best = space_eval(space, best)
        return self.best
