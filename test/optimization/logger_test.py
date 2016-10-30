import os
import unittest

import numpy as np
from sklearn import metrics, datasets
from sqlalchemy import create_engine
from sqlalchemy.orm import clear_mappers

from stacker.optimization.loggers import FileLogger, DBLogger
from stacker.optimization.optimizer.result import OptimizationResult
from stacker.optimization.optimizers import XGBoostOptimizer
from stacker.optimization.scorer import Scorer
from stacker.optimization.task import Task


class TestProgressSaver(unittest.TestCase):
    def setUp(self):
        os.putenv("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.engine = create_engine('sqlite:///test.db')

    def tearDown(self):
        if os.path.exists("test.db"):
            os.remove("test.db")
        clear_mappers()

    def test_db_insertion(self):
        task = Task("test_task", np.zeros((10, 1)), np.zeros(10), "classification", 0.25, 5)
        logger = DBLogger(task, self.engine)
        test_result = OptimizationResult(task.name, "test_model", {"a": 1}, 0.5, "scorer", "cv", [0, 1, 1], 42)
        logger.save(test_result)

    def test_db_logger(self):
        X, y = datasets.make_classification(random_state=42)
        task = Task("class_split", X, y, "classification", test_size=0.1, random_state=42)
        scorer = Scorer("auc_error", lambda y_pred, y_true: 1 - metrics.roc_auc_score(y_pred, y_true))
        logger = DBLogger(task, self.engine)
        optimizer = XGBoostOptimizer(task, scorer, logger)
        optimizer.start_optimization(max_evals=10)
        self.assertEqual(len(list(logger.load_all_results())), 10)

    def test_file_logger(self):
        X, y = datasets.make_classification(random_state=42)
        task = Task("class_split", X, y, "classification", test_size=0.1, random_state=42)
        scorer = Scorer("auc_error", lambda y_pred, y_true: 1 - metrics.roc_auc_score(y_pred, y_true))
        logger = FileLogger(task)
        optimizer = XGBoostOptimizer(task, scorer, logger)
        optimizer.start_optimization(max_evals=10)
        self.assertEqual(len(list(logger.load_all_results())), 10)
        os.remove(task.name + ".log")

if __name__ == '__main__':
    unittest.main()
