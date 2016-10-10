from sklearn.base import BaseEstimator
from typing import *
import numpy as np

from ..optimization.optimizer.task import Task

class Stacker:
    def __init__(self, task: Task):
        if task.kfold is None:
            raise ValueError("task object should have kfold")
        self.task = task

    def get_prediction(self, model, X):
        if self.task.task == "classification":
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X)


class ModelStacker(Stacker):
    def __init__(self, task: Task, models: List[BaseEstimator]):
        super().__init__(task)
        self.models = models

    def build_stacked_from_models(self):
        X_stacked = np.zeros((self.task.X.shape[0], len(self.models)))
        for train_index, test_index in self.task.kfold:
            X_train, X_test = self.task.X[train_index], self.task.X[test_index]
            y_train, y_test = self.task.y[train_index], self.task.y[test_index]
            for i, model in enumerate(self.models):
                model.fit(X_train, y_train)
                X_stacked[test_index, i] = self.get_prediction(model, X_test)
        return X_stacked


class PredictionStacker(Stacker):
    def __init__(self, task: Task, predictions):
        super().__init__(task)
        self.predictions = predictions

    def build_stacked_from_predictions(self):
        X_stacked = np.zeros((self.task.X.shape[0], len(self.predictions)))
        for fold, (train_index, test_index) in enumerate(self.task.kfold):
            for i, preds in enumerate(self.predictions):
                X_stacked[test_index, i] = preds[fold]
        return X_stacked
