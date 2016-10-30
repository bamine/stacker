from ..optimizer.result import OptimizationResult
from ..task import Task


class OptimizationLogger:
    def __init__(self, task: Task):
        self.task = task

    def save(self, test_result: OptimizationResult):
        return NotImplemented

    def load_all_results(self):
        return NotImplemented


