import json
from datetime import datetime

from sqlalchemy import MetaData
from sqlalchemy import Table, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import mapper
from sqlalchemy.orm import scoped_session, sessionmaker

from ..optimization.task import Task
from .logging.optimization_logger import OptimizationLogger
from .optimizer.result import OptimizationResult


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
                    d["task"],
                    d["model"],
                    d["parameters"],
                    d["score"],
                    d["scorer_name"],
                    d["validation_method"],
                    d["predictions"],
                    d["random_state"])
                yield opt

class DBLogger(OptimizationLogger):
    class OptimizationResultLog:
        def __init__(self,
                     task: str,
                     date_time: datetime,
                     model: str,
                     parameters: str,
                     score: float,
                     scorer_name: str,
                     validation_method: str,
                     predictions: str,
                     random_state: int):
            self.task = task
            self.date_time = date_time
            self.model = model
            self.parameters = parameters
            self.score = score
            self.scorer_name = scorer_name
            self.validation_method = validation_method
            self.predictions = predictions
            self.random_state = random_state

    def __init__(self, task: Task, engine):
        super().__init__(task)
        self.table_name = task.name + "_opt_table"
        self.engine = engine
        self.session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
        self.initialize()

    def initialize(self):
        metadata = MetaData()
        logs = Table(self.table_name, metadata,
                     Column('task', String, primary_key=True),
                     Column('date_time', DateTime, primary_key=True),
                     Column('model', String),
                     Column('parameters', String),
                     Column('score', Float),
                     Column('scorer_name', String),
                     Column('validation_method', String),
                     Column('predictions', String),
                     Column('random_state', Integer))
        mapper(self.OptimizationResultLog, logs)
        metadata.create_all(bind=self.engine)

    def save(self, opt_result: OptimizationResult):
        log = self.OptimizationResultLog(task=self.task.name,
                                         date_time=datetime.now(),
                                         model=opt_result.model,
                                         parameters=json.dumps(opt_result.parameters),
                                         score=opt_result.score,
                                         scorer_name=opt_result.scorer_name,
                                         validation_method=opt_result.validation_method,
                                         predictions=json.dumps(opt_result.predictions),
                                         random_state=opt_result.random_state)
        self.session.add(log)
        self.session.commit()

    def load_all_results(self):
        logs = self.session.query(self.OptimizationResultLog).all()
        return logs

    def table_exist(self):
        return self.engine.dialect.has_table(self.engine, self.OptimizationResultLog)