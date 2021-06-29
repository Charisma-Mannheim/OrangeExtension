import numpy as np

class UnNormalize(object):
    def __init__(self):
        self.mean = None
        self.std = None
        self.normalizedData = None

    def __call__(self, data, mean, std):
        self.mean = mean
        self.std = std
        self.normalizedData = data
        originalData = (self.normalizedData * self.std) + self.mean
        return originalData


class Normalize(object):
    def __init__(self):
        self.mean = None
        self.std = None
        self.originalData = None

    def __call__(self, data, mean, std):
        self.mean = mean
        self.std = std
        self.originalData = data
        normalizedData = (self.originalData - self.mean)/ self.std
        return normalizedData


class Center(object):
    def __init__(self):
        self.mean = None
        self.originalData = None

    def __call__(self, data, mean):
        self.mean = mean
        self.originalData = data
        centeredData = (self.originalData - self.mean)
        return centeredData



class UnCenter(object):
    def __init__(self):
        self.mean = None
        self.centeredData = None

    def __call__(self, data, mean):
        self.mean = mean
        self.centeredData = data
        originalData = (self.centeredData + self.mean)
        return originalData

class DoNothing(object):
    def __init__(self):
        self.data = None

    def __call__(self, data):
        self.data = data
        return self.data