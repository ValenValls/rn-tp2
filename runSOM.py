import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from SOM.model import SOM as SomModel
from utils.dataUtils import get_data

# Dimensions of the SOM grid
m = 20
train_data, Y = get_data("./data/tp2_training_dataset.csv")

total_epochs = 0
model = SomModel(20,850)
for epochs, i in zip([1, 4, 5, 10], range(0,4)):
    total_epochs += epochs
    SOM = model.train_SOM(train_data, epochs=epochs)

# U-matrix
def dist3(p1,p2):
    """
    Square of the Euclidean distance between points p1 and p2
    in 3 dimensions.
    """
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2

udm=np.zeros((m-2,m-2))    # initiaize U-matrix with elements set to 0

for i in range(1,m-1):        # loops over the neurons in the grid
    for j in range(1,m-1):
        udm[i-1][j-1]=np.sqrt(dist3(SOM[i][j],SOM[i][j+1])+dist3(SOM[i][j],SOM[i][j-1])+
                            dist3(SOM[i][j],SOM[i+1][j])+dist3(SOM[i][j],SOM[i-1][j]))



fig = plt.figure(figsize=(4,4),dpi=120)
axes1 = fig.add_subplot(111, projection="3d")
ax = fig.gca()

xx_1 = np.arange(1, m-1, 1)
xx_2 = np.arange(1, m-1, 1)

x_1, x_2 = np.meshgrid(xx_1, xx_2)

Z=np.array([[udm[i][j] for i in range(m-2)] for j in range(m-2)])

ax.set_zlim(0,.5)

ax.plot_surface(x_1,x_2, Z, cmap=cm.gray)

plt.xlabel('$i$',fontsize=11)
plt.ylabel('$j$',fontsize=11)
plt.title("U-matrix",fontsize=11)
# plt.show()


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


