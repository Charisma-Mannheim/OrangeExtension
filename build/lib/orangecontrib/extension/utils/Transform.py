import numpy as np

class Normalize(object):
    def __init__(self):
        self.mean = None
        self.std = None
        self.originalData = None
        self.normalizedData = None

    def fit(self, data):
        self.originalData = data
        self.mean = np.mean(self.originalData,0)
        self.std = np.std(self.originalData,0)


    def fit_transform(self, data):
        self.originalData = data
        self.mean = np.mean(self.originalData,0)
        self.std = np.std(self.originalData,0)
        self.normalizedData = (self.originalData - self.mean)/ self.std
        return self.normalizedData

    def transform(self, data):
        self.normalizedData = (data - self.mean)/ self.std
        return self.normalizedData

    def inverse_transform(self, data, index=None):
        if index is not None:
            originalData = (data * self.std[index]) + self.mean[index]
        else:
            originalData = (data * self.std) + self.mean
        return originalData


class Center(object):
    def __init__(self):
        self.mean = None
        self.originalData = None
        self.centeredData = None

    def fit(self, data):
        self.originalData = data
        self.mean = np.mean(self.originalData,0)

    def fit_transform(self, data):
        self.originalData = data
        self.mean = np.mean(self.originalData, 0)
        self.centeredData = (self.originalData - self.mean)
        return self.centeredData

    def transform(self, data):
        self.centeredData = (data - self.mean)
        return self.centeredData

    def inverse_transform(self, data, index=None):
        if index is not None:
            originalData = (data + self.mean[index])
        else:
            originalData = (data + self.mean)
        return originalData


class DoNothing(object):
    def __init__(self):
        self.data = None

    def __call__(self, data):
        self.data = data
        return self.data