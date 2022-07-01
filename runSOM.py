import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from SOM.model import SOM as SomModel
from utils.dataUtils import get_data, plotSOM, u_matrix

# Dimensions of the SOM grid
m = 20
train_data, Y = get_data("./data/tp2_training_dataset.csv")

total_epochs = 0
model = SomModel(850, 20)
for epochs, i in zip([1, 4, 5, 10], range(0,4)):
    total_epochs += epochs
    SOM = model.train(train_data, epochs=epochs)

udm = u_matrix(SOM,m)

plotSOM(udm, m)
plt.show()


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


