import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from SOM.model import SOM as SomModel
from hebbiano.model import Hebbiano
from utils.dataUtils import get_data, plotSOMColorMap, normalize, proportional_separate_train_validation


train_data, Y = get_data("./data/tp2_training_dataset.csv")

# TODO prueba pasando por hebb primero para reducir dimensionalidad
# para usar cambiar train_data por X en la linea 10
# X, _, _ = normalize(X)
#
# hebbianoOja = Hebbiano(850,9)
# hebbianoOja.trainOja(X)
# train_data = hebbianoOja.predict(X)

# Dimensions of the SOM grid
m = 6
X_train, Y_train, X_val, Y_val = proportional_separate_train_validation(train_data,Y)
total_epochs = 0
model = SomModel(train_data.shape[1], m)
SOM = model.train(X_train, learn_rate=.3, radius_sq=3, epochs=16, graph=True, Y=Y_train)
model.categorize_and_map(X_val, Y_val, learn_rate=.3, radius=3, epochs=16)

# plt.show()

# c= model.categorize(train_data, Y)
# print(c)
# plotSOMColorMap(c)

