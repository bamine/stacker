import sqlite3
import json

from stacker.optimization.optimizer.task import Task
from stacker.optimization.optimizer.result import OptimizationResult


class OptimizationLogger:
    def __init__(self, task: Task):
        self.task = Task

    def save(self, test_result: OptimizationResult):
        return NotImplemented

    def load_all_results(self):
        return NotImplemented


class VoidLogger(OptimizationLogger):
    def save(self, test_result: OptimizationResult):
        pass


class FileLogger(OptimizationLogger):
    def __init__(self, task: Task):
        super().__init__(task)
        self.file_name = task.name + ".log"

    def save(self, opt_result: OptimizationResult):
        with open(self.file_name, "a") as f:
            f.write(json.dumps(opt_result.__dict__))
            f.write("\n")

    def load_all_results(self):
        with open(self.file_name) as f:
            for line in f:
                d = json.loads(line)
                opt = OptimizationResult(
                    d["model"],
                    d["parameters"],
                    d["score"],
                    d["scorer_name"],
                    d["validation_method"],
                    d["predictions"],
                    d["random_state"])
                yield opt



class DBLogger(OptimizationLogger):
    def __init__(self, task: Task):
        super().__init__(task)
        self.db_name = task.name
        self.hp_opt_table = task.name + "_hp_opt_table"
        self.conn = sqlite3.connect(self.db_name)
        self.init_table(self.hp_opt_table)

    def init_table(self, table_name):
        self.conn.execute("""
            create table if not exists {0}(
            model text,
            parameters text,
            scorer_name text,
            score double,
            validation_method text,
            predictions text,
            random_state integer)
        """.format(table_name))
        self.conn.commit()

    def save(self, opt_result: OptimizationResult):
        c = self.conn.cursor()
        c.execute("INSERT into {0} VALUES (?, ?, ?, ?, ?, ?, ?)".format(self.hp_opt_table),
                  (
                      opt_result.model,
                      json.dumps(opt_result.parameters),
                      opt_result.scorer_name,
                      opt_result.score,
                      opt_result.validation_method,
                      json.dumps(opt_result.predictions),
                      opt_result.random_state
                  ))
        self.conn.commit()

    def load_all_results(self):
        c = self.conn.cursor()
        for row in c.execute("SELECT * FROM {0}".format(self.hp_opt_table)):
            model, parameters, scorer_name, score, validation_method, predictions, random_state = row
            yield OptimizationResult(
                model,
                json.loads(parameters),
                scorer_name,
                score,
                validation_method,
                json.loads(predictions),
                random_state)

    def table_exist(self, table_name):
        c = self.conn.cursor()
        tables = c.execute("SELECT name FROM sqlite_master WHERE type = 'table' and name='{0}'".format(table_name))
        return tables.fetchone()


