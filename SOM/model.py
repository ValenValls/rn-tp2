import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches


class SOM:

    def __init__(self, n=None, m=None):
        ## n es la cantidad de palabras, m el tamaño de la matriz de pesos de cada celula.
        rand = np.random.RandomState(0)
        self.n = n
        self.m = m
        if n and m:
            self.SOM = rand.randn(m, m, n) *0.1

    def export_model(self, filename):
        # El formato del archivo seria
        # N = Cantidad de palabras
        # M = Donde MxM es el tamaño de la matriz de pesos de cada palabra
        # A continuacion, la matriz de pesos de (m, m, n)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"{self.n}\n")
            file.write(f"{self.m}\n")
            for layer in self.SOM:
                np.savetxt(file, layer, fmt='%.6f')

    def import_model(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            file_rows = file.readlines()
        self.n = int(file_rows[0])
        self.m = int(file_rows[1])
        mat = []
        for l1 in range(2, 2+self.m):
            mat.append(np.asarray(np.matrix((';').join([row[:-1] for row in file_rows[l1: (l1 + self.m)]]))))
        self.SOM = np.stack(mat)
        assert self.SOM.shape == (self.m, self.m, self.n), self.SOM.shape


    # Return the (g,h) index of the BMU in the grid
    def find_BMU(self, x):
        distSq = (np.square(self.SOM - x)).sum(axis=2)
        return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)


    # Update the weights of the SOM cells when given a single training example
    # and the model parameters along with BMU coordinates as a tuple
    def update_weights(self, train_ex, learn_rate, radius_sq,
                       BMU_coord, step=3):
        g, h = BMU_coord
        # if radius is close to zero then only BMU is changed
        if radius_sq < 1e-3:
            self.SOM[g, h, :] += learn_rate * (train_ex - self.SOM[g, h, :])
            return self.SOM
        # Change all cells in a small neighborhood of BMU
        for i in range(max(0, g - step), min(self.SOM.shape[0], g + step)):
            for j in range(max(0, h - step), min(self.SOM.shape[1], h + step)):
                dist_sq = np.square(i - g) + np.square(j - h)
                dist_func = np.exp(-dist_sq / 2 / radius_sq)
                self.SOM[i, j, :] += learn_rate * dist_func * (train_ex - self.SOM[i, j, :])
        return self.SOM


    # Main routine for training an SOM. It requires an initialized SOM grid
    # or a partially trained grid as parameter
    def train(self, train_data, learn_rate=.99, radius_sq=10,
              lr_decay=.1, radius_decay=.1, epochs=10, graph=False, Y=None, fn='.png'):
        learn_rate_0 = learn_rate
        radius_0 = radius_sq
        if graph:
            X = train_data.copy()
            scale = math.ceil(epochs/8)
            to_graph = [0]
            for i in range(0, 6):
                to_graph.append(to_graph[i] + scale)
            to_graph.append(epochs-1)
            fig, ax = plt.subplots(
                nrows=2, ncols=4, figsize=(15, 5),
                subplot_kw=dict(xticks=[], yticks=[]))
        else:
            graph=[]

        for epoch in np.arange(0, epochs):
            np.random.shuffle(train_data)
            for train_ex in train_data:
                g, h = self.find_BMU(train_ex)
                self.SOM = self.update_weights(train_ex,
                                     learn_rate, radius_sq, (g, h))
            # Update learning rate and radius
            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)
            if epoch in to_graph:
                idx = to_graph.index(epoch)
                x = idx % 4
                y = idx // 4
                categorized = self.categorize(X, Y)
                current_ax = ax[y][x]
                self.plot_ax(categorized, current_ax, epoch)

        if graph:
            fig.subplots_adjust(hspace=.3)
            fig.suptitle(f"Mapeo de Características - lr: {learn_rate_0} - radio: {radius_0} ", fontsize=14)
            fig.savefig(f"SOM_lr_{learn_rate_0}_radio_{radius_0}_epochs_{epochs}_{fn}")
            fig.show()
            # fig.close()
        return self.SOM

    def plot_ax(self, categorized, current_ax, epoch, title='Epochs = ', cmap='Paired', legend='Categoria'):
        im = current_ax.imshow(categorized, cmap=cmap)
        title = f'{title} ' + str(epoch + 1)
        current_ax.title.set_text(title)
        values = np.unique(categorized.ravel())
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=values[i])) for i in
                   range(len(values))]
        # Major ticks
        current_ax.set_xticks(np.arange(0, self.m, 1))
        current_ax.set_yticks(np.arange(0, self.m, 1))
        # Labels for major ticks
        current_ax.set_xticklabels(np.arange(1, self.m + 1, 1))
        current_ax.set_yticklabels(np.arange(1, self.m + 1, 1))
        # Minor ticks
        current_ax.set_xticks(np.arange(-.5, self.m, 1), minor=True)
        current_ax.set_yticks(np.arange(-.5, self.m, 1), minor=True)
        # Gridlines based on minor ticks
        current_ax.grid(which='minor', color='black', linestyle='--', linewidth=1)
        current_ax.legend(title=legend, handles=patches, bbox_to_anchor=(1.05, .95), loc=2,
                          borderaxespad=0., labelspacing=0, handleheight=0.5,
                          fontsize='x-small')

    ## U-MATRIX
    def dist3(self, p1, p2):
        """
        Square of the Euclidean distance between points p1 and p2
        in 3 dimensions.
        """
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2

    def u_matrix(self):
        udm = np.zeros((self.m - 2, self.m - 2))  # initiaize U-matrix with elements set to 0
        SOM = self.SOM
        for i in range(1, self.m - 1):  # loops over the neurons in the grid
            for j in range(1, self.m - 1):
                udm[i - 1][j - 1] = np.sqrt(self.dist3(SOM[i][j], SOM[i][j + 1]) + self.dist3(SOM[i][j], SOM[i][j - 1]) +
                                            self.dist3(SOM[i][j], SOM[i + 1][j]) + self.dist3(SOM[i][j], SOM[i - 1][j]))
        return udm

    #Tomo un modelo entrenado, conjunto de entrenamiento/validacion X y etiquetas Y
    #Retorno una matriz de mxm(mismas 2 primeras dimensiones del SOM) donde en cada
    #posicion esta el numero de categoria del que mas ejemplos cayeron en esa posicion
    #del SOM al buscar su BMU
    #todo No se si sera el nombre mas adecuado
    def categorize(self, X, Y):
        #En cada posicion i,j,k de cat voy a contar cuantos documentos entran en la posicion i,j
        #del SOM de la categoria k+1
        #La tercer coordenada es 10 ya que el 0 va a representar que ningun documento entro ahi
        cat = self.map_per_cat(X, Y)
        totalCat= np.argmax(cat, axis=2)
        return totalCat

    def map_per_cat(self, X, Y):
        cat = np.zeros((self.SOM.shape[0], self.SOM.shape[1], 10))
        for c, x in enumerate(X):
            i, j = self.find_BMU(x)
            cat[i][j][Y[c]] += 1
        return cat