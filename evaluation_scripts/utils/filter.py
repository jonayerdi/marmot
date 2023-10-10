import abc
from collections import deque

class Filter(abc.ABC):
    def __init__(self) -> None:
        self.reset()
    @abc.abstractmethod
    def compute(self):
        raise Exception('Abstract method')
    @abc.abstractmethod
    def reset(self):
        raise Exception('Abstract method')
    @abc.abstractmethod
    def next(self, datapoint: float) -> float:
        raise Exception('Abstract method')

class SimpleARFilter(Filter):
    def __init__(self, coefficients=[1 / (n + 1) for n in range(10)], scale=None) -> None:
        self.coefficients = coefficients
        if scale is None:
            scale = sum(coefficients)
        self.scale = scale
        super().__init__()
    def compute(self):
        return sum(map(lambda v: v[0] * v[1], zip(self.coefficients, self.data))) / self.scale
    def reset(self):
        self.data = deque((0.0 for _ in self.coefficients))
    def next(self, datapoint: float) -> float:
        self.data.pop()
        self.data.appendleft(datapoint)
        return self

def init_filter(name, **kwargs):
    return {
        f.__name__: f for f in (SimpleARFilter,)
    }[name](**kwargs)
