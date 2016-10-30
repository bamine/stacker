from abc import ABC, abstractmethod
from hyperopt import hp


class ParameterDistribution(ABC):
    @abstractmethod
    def hyperopt(self):
        pass


class Choice(ParameterDistribution):
    def __init__(self, label, choices):
        self.label = label
        self.options = choices

    def hyperopt(self):
        return hp.choice(self.label.name, self.options)


class RandInt(ParameterDistribution):
    def __init__(self, label, upper):
        self.label = label
        self.upper = upper

    def hyperopt(self):
        return hp.randint(self.label.name, self.upper)


class Uniform(ParameterDistribution):
    def __init__(self, label, low, high):
        self.label = label
        self.low = low
        self.high = high

    def hyperopt(self):
        return hp.uniform(self.label.name, self.low, self.high)


class QUniform(ParameterDistribution):
    def __init__(self, label, low, high, q):
        self.label = label
        self.low = low
        self.high = high
        self.q = q

    def hyperopt(self):
        return hp.quniform(self.label.name, self.low, self.high, self.q)


class LogUniform(ParameterDistribution):
    def __init__(self, label, low, high):
        self.label = label
        self.low = low
        self.high = high

    def hyperopt(self):
        return hp.loguniform(self.label.name, self.low, self.high)


class QLogUniform(ParameterDistribution):
    def __init__(self, label, low, high, q):
        self.label = label
        self.low = low
        self.high = high
        self.q = q

    def hyperopt(self):
        return hp.qloguniform(self.label.name, self.low, self.high, self.q)


class Normal(ParameterDistribution):
    def __init__(self, label, mu, sigma):
        self.label = label
        self.mu = mu
        self.sigma = sigma

    def hyperopt(self):
        return hp.normal(self.label.name, self.mu, self.sigma)


class QNormal(ParameterDistribution):
    def __init__(self, label, mu, sigma, q):
        self.label = label
        self.mu = mu
        self.sigma = sigma
        self.q = q

    def hyperopt(self):
        return hp.qnormal(self.label.name, self.mu, self.sigma, self.q)


class LogNormal(ParameterDistribution):
    def __init__(self, label, mu, sigma):
        self.label = label
        self.mu = mu
        self.sigma = sigma

    def hyperopt(self):
        return hp.lognormal(self.label.name, self.mu, self.sigma)


class QLogNormal(ParameterDistribution):
    def __init__(self, label, mu, sigma, q):
        self.label = label
        self.mu = mu
        self.sigma = sigma
        self.q = q

    def hyperopt(self):
        return hp.qlognormal(self.label.name, self.mu, self.sigma, self.q)