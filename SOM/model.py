import os

import numpy as np
import math
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches


class SOM:

    def __init__(self, n=None, m=None):
        ## n es la cantidad de palabras, m el tamaño de la matriz de pesos de cada celula.
        rand = np.random.RandomState(0)
        self.n = n
        self.m = m
        if n and m:
            self.SOM = rand.randn(m, m, n) *0.1
            self.categories = np.zeros((m, m))

    #Metodo que permite variar el m una vez creado el modelo. Resetea los pesos y categorias.
    def change_m_and_reset(self,m):
        rand = np.random.RandomState(0)
        self.m = m
        self.SOM = rand.randn(m, m, self.n) *0.1
        self.categories = np.zeros((m, m))


    def export_model(self, filename, graph=False):
        # El formato del archivo seria
        # N = Cantidad de palabras
        # M = Donde MxM es el tamaño de la matriz de pesos de cada palabra
        # A continuacion, la matriz de pesos de (m, m, n)
        # Por ultimo la matriz de categorias de mxm
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"{self.n}\n")
            file.write(f"{self.m}\n")
            for layer in self.SOM:
                np.savetxt(file, layer, fmt='%.6f')
            np.savetxt(file, self.categories, fmt='%i')
        if graph:
            self.plot_mapeo_categorias()

    def import_model(self, filename, graph=False):
        with open(filename, 'r', encoding='utf-8') as file:
            file_rows = file.readlines()
        self.n = int(file_rows[0])
        self.m = int(file_rows[1])
        mat = []
        for l1 in range(0, self.m):
            start = 2 + l1 * self.m
            end = start + self.m
            mat.append(np.asarray(np.matrix((';').join([row[:-1] for row in file_rows[start: end]]))))
        self.SOM = np.stack(mat)
        self.categories = np.asarray(np.matrix((';').join([row[:-1] for row in file_rows[-(self.m): ]])))
        assert self.SOM.shape == (self.m, self.m, self.n), self.SOM.shape
        assert self.categories.shape == (self.m, self.m), self.categories.shape
        if graph:
            self.plot_mapeo_categorias()


    # Devuelve el indice del BMU en la grilla
    def find_BMU(self, x):
        distSq = (np.square(self.SOM - x)).sum(axis=2)
        return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)


    # Actualizo los pesos de cada posicion del SOM dado una sola instancia del conjunto de entrada,
    # los parametros del modelo y las coordenadas del BMU como tupla
    def update_weights(self, train_ex, learn_rate, radius_sq,
                       BMU_coord, step=3):
        g, h = BMU_coord
        # Si el radio es cercano a 0 solo se modifica el BMU
        if radius_sq < 1e-3:
            self.SOM[g, h, :] += learn_rate * (train_ex - self.SOM[g, h, :])
            return self.SOM
        # Modifico los pesos en un vecindario chico alrededor del BMU
        for i in range(max(0, g - step), min(self.SOM.shape[0], g + step)):
            for j in range(max(0, h - step), min(self.SOM.shape[1], h + step)):
                dist_sq = np.square(i - g) + np.square(j - h)
                dist_func = np.exp(-dist_sq / 2 / radius_sq)
                self.SOM[i, j, :] += learn_rate * dist_func * (train_ex - self.SOM[i, j, :])
        return self.SOM


    # Algoritmo de entrenamiento del SOM. Toma un tensor del SOM inicializado o
    #un SOM parcialmente entrenado
    def train(self, train_data, learn_rate=.99, radius_sq=10,
              lr_decay=.1, radius_decay=.1, epochs=10, graph=False, Y=[], fn='SOM.png', path=''):
        learn_rate_0 = learn_rate
        radius_0 = radius_sq
        to_graph = []
        X = train_data.copy()
        if graph:
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
            np.random.shuffle(X)
            for train_ex in X:
                g, h = self.find_BMU(train_ex)
                self.SOM = self.update_weights(train_ex,
                                     learn_rate, radius_sq, (g, h))
            # Actualizo learning rate y radius
            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)
            if epoch in to_graph:
                idx = to_graph.index(epoch)
                x = idx % 4
                y = idx // 4
                categorized = self.categorize_grid(train_data, Y)
                current_ax = ax[y][x]
                self.plot_ax(categorized, current_ax, epoch)

        if graph:
            fig.subplots_adjust(hspace=.3)
            fig.suptitle(f"Mapeo de Características - lr: {learn_rate_0} - radio: {radius_0} ", fontsize=14)
            file = os.path.join(path,f"{fn}_train.png")
            fig.savefig(file)

        if len(Y) > 0:
            self.categories = self.categorize_grid(train_data, Y)
            acc, _, _ = self.accuracy(train_data,Y)

        return self.SOM, acc


    def categorize(self, X):
        pred = []
        for doc in X:
            i, j = self.find_BMU(doc)
            pred.append(self.categories[i, j])
        return pred

    #Tomo un modelo entrenado, conjunto de entrenamiento/validacion X y etiquetas Y
    #Retorno una matriz de mxm(mismas 2 primeras dimensiones del SOM) donde en cada
    #posicion esta el numero de categoria del que mas ejemplos cayeron en esa posicion
    #del SOM al buscar su BMU
    def categorize_grid(self, X, Y):
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

    def accuracy(self, X, Y):
        total =np.zeros(10)
        accurate = np.zeros(10)
        for doc, cat in zip(X,Y):
            i, j = self.find_BMU(doc)
            predicted = self.categories[i,j]
            total[cat] +=1
            if predicted == cat:
                accurate[cat] += 1
        acc = sum(accurate)/sum(total)
        return acc, total, accurate

    ### METODOS PARA GRAFICAR
    def plot_mapeo_categorias(self, fn='mapa_caracteristicas.png', path=''):
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(15, 20),
            subplot_kw=dict(xticks=[], yticks=[]))
        self.plot_ax(self.categories, ax, title='Mapa de categorias recuperado ', cmap='Paired', legend='Categoria')
        file = os.path.join(path, fn)
        fig.savefig(file)

    def plot_ax(self, categorized, current_ax, epoch, title='Epochs = ', cmap='Paired', legend='Categoria'):
        im = current_ax.imshow(categorized, cmap=cmap)
        title = (f'{title} ' + str(epoch + 1) ) if epoch else title
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

    def categorize_and_map(self, X, Y, fn='validation.png', title='Clasificación del set de validación', path=''):
        mat = self.map_per_cat(X, Y)
        # fig = plt.figure(2)
        fig, ax = plt.subplots(
            nrows=3, ncols=3, figsize=(15, 20),
            subplot_kw=dict(xticks=[], yticks=[]))
        for idx in range(0, 9):
            x = idx % 3
            y = idx // 3
            current_ax = ax[y][x]
            self.plot_ax(mat[:, :, idx + 1], current_ax, idx, cmap='gray_r', title='Categoria ', legend='#Docs')
        fig.subplots_adjust(hspace=.3, wspace=.3)
        fig.suptitle(f"{title} ", fontsize=14)
        file = os.path.join(path, f"{fn}_classify.png")
        fig.savefig(file)