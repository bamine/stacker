import sqlite3
import json

from stacker.optimization.optimizer.task import Task
from stacker.optimization.optimizer.result import OptimizationResult


class Logger:
    def __init__(self, task: Task):
        self.task = Task

    def save(self, test_result: OptimizationResult):
        return NotImplemented


class FileLogger(Logger):
    def __init__(self, task: Task):
        super(FileLogger).__init__(task)
        self.file_name = task.name + ".log"

    def save(self, test_result: OptimizationResult):
        with open(self.file_name, "a") as f:
            f.write(json.dumps(test_result.__dict__))


class DBLogger(Logger):
    def __init__(self, task: Task):
        super(DBLogger).__init__(task)
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

    def save(self, test_result: OptimizationResult):
        c = self.conn.cursor()
        c.execute("INSERT into {0} VALUES (?, ?, ?, ?, ?, ?, ?)".format(self.hp_opt_table),
                  (
                      test_result.model,
                      json.dumps(test_result.parameters),
                      test_result.scorer_name,
                      test_result.score,
                      test_result.validation_method,
                      json.dumps(test_result.predictions),
                      test_result.random_state
                  ))
        self.conn.commit()

    def table_exist(self, table_name):
        c = self.conn.cursor()
        tables = c.execute("SELECT name FROM sqlite_master WHERE type = 'table' and name='{0}'".format(table_name))
        return tables.fetchone()


