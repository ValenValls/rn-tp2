import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from SOM.model import SOM as SomModel
from utils.dataUtils import get_data, plotSOM, plotSOMColorMap

# Dimensions of the SOM grid
m = 6
train_data, Y = get_data("./data/tp2_training_dataset.csv")

total_epochs = 0
model = SomModel(850, m)
SOM = model.train(train_data, epochs=20, graph=True, Y=Y)
# fig, ax = plt.subplots(
#     nrows=2, ncols=4, figsize=(15, 3.5),
#     subplot_kw=dict(xticks=[], yticks=[]))
# for epochs, i in zip([1, 4, 5, 10, 2, 2, 2, 2], range(0,8)):
#     total_epochs += epochs
#     x= i % 4
#     y = i // 4
#     SOM = model.train(train_data, epochs=epochs)
#     ax[y][x].imshow(model.categorize(train_data, Y), cmap='Pastel1')
#     ax[y][x].title.set_text('Epochs = ' + str(total_epochs))

# plt.show()

# c= model.categorize(train_data, Y)
# print(c)
# plotSOMColorMap(c)
#udm = u_matrix(SOM,m)

# plotSOM(udm, m)
#
# tad = np.zeros((m, m))
#
# for i in range(m):
#     for j in range(m):
#         tad[i][j] = dist3(train_data[1], SOM[i][j])
#
# ind_m = np.argmin(tad)  # winner
# in_x = ind_m // m
# in_y = ind_m % m
#
# da = np.sqrt(tad[in_x][in_y])
#
# ax.plot([in_x], [in_y], [da], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6)
# # ax.scatter3D(in_x, in_y, da, c='red')
# plt.show()
# print("Closest neuron grid indices: (", in_x, ",", in_y, ")")
# print("Distance: ", np.round(da, 3))


