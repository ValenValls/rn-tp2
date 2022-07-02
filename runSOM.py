import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from SOM.model import SOM as SomModel
from hebbiano.model import Hebbiano
from utils.dataUtils import get_data, plotSOM, plotSOMColorMap, normalize, proportional_separate_train_validation

# Dimensions of the SOM grid
m = 6
train_data, Y = get_data("./data/tp2_training_dataset.csv")

# TODO prueba pasando por hebb primero para reducir dimensionalidad
# para usar cambiar train_data por X en la linea 10
# X, _, _ = normalize(X)
#
# hebbianoOja = Hebbiano(850,9)
# hebbianoOja.trainOja(X)
# train_data = hebbianoOja.predict(X)

X_train, Y_train, X_val, Y_val = proportional_separate_train_validation(train_data,Y)

total_epochs = 0
model = SomModel(train_data.shape[1], m)
SOM = model.train(X_train, learn_rate=.3, radius_sq=3, epochs=16, graph=True, Y=Y_train)

fig = plt.figure(2)
fig, ax = plt.subplots(
    nrows=3, ncols=3, figsize=(15, 20),
    subplot_kw=dict(xticks=[], yticks=[]))
mat = model.map_per_cat(X_val,Y_val)
for idx in range(0,9):
    x = idx % 3
    y = idx // 3
    current_ax = ax[y][x]
    model.plot_ax(mat[:,:,idx+1], current_ax, idx, cmap='gray_r', title='Categoria ', legend='#Docs')
# model.plot_ax(,ax,epoch=20, title="Validation")
fig.savefig('otraprueba.png')
fig.show()



# plt.show()

# c= model.categorize(train_data, Y)
# print(c)
# plotSOMColorMap(c)

