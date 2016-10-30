import os
import unittest
from sklearn import datasets, metrics

from stacker.optimization.optimizer.task import Task
from stacker.optimization.optimizer.scorer import Scorer
from stacker.optimization.optimizers import XGBoostOptimizer, RandomForestOptimizer


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        os.putenv("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.X_class, self.y_class = datasets.make_classification(random_state=42)
        self.X_reg, self.y_reg = datasets.make_regression(random_state=42)
        self.classification_optimizers = [XGBoostOptimizer, RandomForestOptimizer]
        self.regression_optimizers = [XGBoostOptimizer, RandomForestOptimizer]
        self.class_scorer = Scorer("auc_error", lambda y_pred, y_true: 1 - metrics.roc_auc_score(y_pred, y_true))
        self.reg_scorer = Scorer("mse", metrics.mean_squared_error)

        self.classification_task_split = \
            Task("class_split", self.X_class, self.y_class, "classification", test_size=0.1, random_state=42)
        self.regression_task_split = \
            Task("reg_split", self.X_class, self.y_class, "regression", test_size=0.1, random_state=42)

        self.classification_task_cv = \
            Task("class_cv", self.X_reg, self.y_reg, "classification", cv=5, random_state=42)
        self.regression_task_cv = \
            Task("reg_cv", self.X_reg, self.y_reg, "regression", cv=5, random_state=42)

    def test_xgboost(self):
        xgboost_optimizer = XGBoostOptimizer(self.classification_task_split, self.class_scorer)
        class_result_split = xgboost_optimizer.score({"max_depth": 5, "n_estimators": 100})

        xgboost_optimizer = XGBoostOptimizer(self.classification_task_split, self.class_scorer)
        class_result_cv = xgboost_optimizer.score({"max_depth": 5, "n_estimators": 100})

        xgboost_optimizer = XGBoostOptimizer(self.classification_task_split, self.reg_scorer)
        reg_result_split = xgboost_optimizer.score({"max_depth": 5, "n_estimators": 100})

        xgboost_optimizer = XGBoostOptimizer(self.classification_task_split, self.reg_scorer)
        reg_result_cv = xgboost_optimizer.score({"max_depth": 5, "n_estimators": 100})

        self.assertLess(class_result_split['loss'], 0.1)
        self.assertLess(class_result_cv['loss'], 0.1)
        self.assertLess(reg_result_split['loss'], 1)
        self.assertLess(reg_result_cv['loss'], 1)

    def test_rf(self):
        rf_optimizer = XGBoostOptimizer(self.classification_task_split, self.class_scorer)
        class_result_split = rf_optimizer.score({"max_depth": 5, "n_estimators": 100})

        rf_optimizer = XGBoostOptimizer(self.classification_task_split, self.class_scorer)
        class_result_cv = rf_optimizer.score({"max_depth": 5, "n_estimators": 100})

        rf_optimizer = XGBoostOptimizer(self.classification_task_split, self.reg_scorer)
        reg_result_split = rf_optimizer.score({"max_depth": 5, "n_estimators": 100})

        rf_optimizer = XGBoostOptimizer(self.classification_task_split, self.reg_scorer)
        reg_result_cv = rf_optimizer.score({"max_depth": 5, "n_estimators": 100})

        self.assertLess(class_result_split['loss'], 0.1)
        self.assertLess(class_result_cv['loss'], 0.1)
        self.assertLess(reg_result_split['loss'], 0.1)
        self.assertLess(reg_result_cv['loss'], 0.1)

    def test_classification_optimization(self):
        for optimizer in self.classification_optimizers:
            opt = optimizer(self.classification_task_split, self.class_scorer)
            print(opt)
            best = opt.start_optimization(max_evals=5)
            result = opt.score(best)
            self.assertLess(result['loss'], 0.1)
            print("done !")

    def test_regression_optimization(self):
        for optimizer in self.regression_optimizers:
            opt = optimizer(self.regression_task_split, self.class_scorer)
            best = opt.start_optimization(max_evals=5)
            result = opt.score(best)
            self.assertLess(result['loss'], 0.1)

if __name__ == '__main__':
    unittest.main()
