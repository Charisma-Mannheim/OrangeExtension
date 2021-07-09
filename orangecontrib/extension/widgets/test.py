from Orange.data import Table
from sklearn import preprocessing
import numpy as np

encoder = preprocessing.LabelEncoder()
data = Table("L:/Promotion/Orange/AddOnsAKWeller/Datensaetze/winequality-white.csv")
b = data.Y.shape[0]
print(b)
data.Y = data.Y.reshape((b,))
print(data.Y.shape)
data.Y = encoder.fit_transform(data.Y)
encoder.classes_