""" Base EvalModule Classes. """
from collections import abstractmethod
from abc import ABC

class EvalModule(ABC):
    """ abstract class for Evaluation Module. """

    @abstractmethod
    @property
    def score(self):
        raise NotImplementedError('score property is not implemented')

    @abstractmethod
    def accumulate(self):
        raise NotImplementedError('accumulate method is not implemented')

    @abstractmethod
    def evaluate(self, results):
        raise NotImplementedError('evaluate method is not implemented.')
