from stacker.optimization.optimizer.task import Task
from stacker.optimization.optimizer.result import OptimizationResult
from stacker.optimization.logging.logger import ProgressSaver

import numpy as np
import unittest


class TestProgressSaver(unittest.TestCase):

    def setUp(self):
        self.task = Task("test_task", np.zeros((10, 1)), np.zeros(10), "classification", 0.25, 5)
        self.saver = ProgressSaver(self.task)

    def test_db_creation(self):
        self.assertTrue(self.saver.table_exist(self.saver.hp_opt_table))

    def test_db_insertion(self):
        test_result = OptimizationResult("test_model", {"a": 1}, 0.5, "scorer", "cv", [0, 1, 1], 42)
        self.saver.save(test_result)

if __name__ == '__main__':
    unittest.main()
