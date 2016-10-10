from stacker.optimization.spaces.space import Space
from stacker.optimization.parameters.xgboost import *
from stacker.optimization.parameters.parameter_distribution import *


def general_space():
    space = {
        max_depth: Uniform(max_depth, 3, 50),
        learning_rate: QUniform(learning_rate, 0.0001, 0.5, 0.0001),
        n_estimators: Uniform(n_estimators, 10, 5000),
        gamma: Uniform(gamma, 0, 10),
        min_child_weight: Uniform(min_child_weight, 0.1, 10),
        max_delta_step: Uniform(max_delta_step, 0, 10),
        subsample: Uniform(subsample, 0.1, 1),
        colsample_bytree: Uniform(colsample_bytree, 0.1, 1),
        colsample_bylevel: Uniform(colsample_bylevel, 0.5, 1),
        reg_alpha: LogUniform(reg_alpha, -5, 5),
        reg_lambda: LogUniform(reg_lambda, -5, 5),
        scale_pos_weight: Uniform(scale_pos_weight, 0, 100),
        base_score: Uniform(base_score, 0, 1),
    }
    return Space(space)


def classification_space():
    space = general_space()
    space.fix_parameters([reg_alpha, reg_lambda])
    return space
