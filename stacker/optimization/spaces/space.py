from typing import *

from stacker.optimization.parameters.parameter_distribution import ParameterDistribution
from stacker.optimization.parameters.parameter import Parameter


class Space:
    def __init__(self, parameter_space: Dict[Parameter, ParameterDistribution]):
        self.parameter_space = parameter_space

    def fix_parameter(self, parameter: Parameter):
        self.parameter_space.pop(parameter, None)

    def fix_parameters(self, parameters: List[Parameter]):
        for p in parameters:
            self.fix_parameter(p)

    def set_parameter_space(self, parameter: Parameter, space: ParameterDistribution):
        self.parameter_space[parameter] = space

    def hyperopt(self):
        return dict([(k.name, v.hyperopt()) for k, v in self.parameter_space.items()])
