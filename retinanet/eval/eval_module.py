""" Base EvalModule Classes. """
from abc import ABC, abstractmethod


class EvalModule(ABC):
    """ abstract class for Evaluation Module. """

    @property
    @abstractmethod
    def scores(self):
        raise NotImplementedError('`scores` property is not implemented')

    @abstractmethod
    def accumulate(self):
        raise NotImplementedError('`accumulate` method is not implemented')

    @abstractmethod
    def evaluate(self, results):
        raise NotImplementedError('`evaluate` method is not implemented')
