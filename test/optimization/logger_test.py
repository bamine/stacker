from stacker.optimization.models import XGBoostOptimizer
from stacker.optimization.optimizer.scorer import Scorer
from stacker.optimization.optimizer.task import Task
from stacker.optimization.optimizer.result import OptimizationResult
from stacker.optimization.logging.optimization_logger import DBLogger, FileLogger

from sklearn import metrics, datasets
import numpy as np
import unittest
import os


class TestProgressSaver(unittest.TestCase):
    def setUp(self):
        os.putenv("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.task = Task("test_task", np.zeros((10, 1)), np.zeros(10), "classification", 0.25, 5)
        self.saver = DBLogger(self.task)

    def tearDown(self):
        os.remove(self.saver.db_name)

    def test_db_creation(self):
        self.assertTrue(self.saver.table_exist(self.saver.hp_opt_table))

    def test_db_insertion(self):
        test_result = OptimizationResult("test_model", {"a": 1}, 0.5, "scorer", "cv", [0, 1, 1], 42)
        self.saver.save(test_result)

    def test_db_logger(self):
        X, y = datasets.make_classification(random_state=42)
        task = Task("class_split", X, y, "classification", test_size=0.1, random_state=42)
        scorer = Scorer("auc_error", lambda y_pred, y_true: 1 - metrics.roc_auc_score(y_pred, y_true))
        logger = DBLogger(task)
        optimizer = XGBoostOptimizer(task, scorer, logger)
        optimizer.start_optimization(max_evals=10)
        self.assertEqual(len(list(logger.load_all_results())), 10)
        os.remove(logger.db_name)

    def test_file_logger(self):
        X, y = datasets.make_classification(random_state=42)
        task = Task("class_split", X, y, "classification", test_size=0.1, random_state=42)
        scorer = Scorer("auc_error", lambda y_pred, y_true: 1 - metrics.roc_auc_score(y_pred, y_true))
        logger = FileLogger(task)
        optimizer = XGBoostOptimizer(task, scorer, logger)
        optimizer.start_optimization(max_evals=10)
        self.assertEqual(len(list(logger.load_all_results())), 10)
        os.remove(logger.file_name)

if __name__ == '__main__':
    unittest.main()
