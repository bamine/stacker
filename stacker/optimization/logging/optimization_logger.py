import json

from stacker.optimization.optimizer.task import Task
from stacker.optimization.optimizer.result import OptimizationResult


class OptimizationLogger:
    def __init__(self, task: Task):
        self.task = task

    def save(self, test_result: OptimizationResult):
        return NotImplemented

    def load_all_results(self):
        return NotImplemented


