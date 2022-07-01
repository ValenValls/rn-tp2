import numpy as np


class SOM:

    def __init__(self, n, m):
        ## n es la cantidad de palabras, m el tamaño de la matriz de pesos de cada celula.
        rand = np.random.RandomState(0)
        self.SOM = rand.randn(m, m, n) * 0.01

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
    def train(self, train_data, learn_rate=.1, radius_sq=1,
              lr_decay=.1, radius_decay=.1, epochs=10):
        learn_rate_0 = learn_rate
        radius_0 = radius_sq
        for epoch in np.arange(0, epochs):
            np.random.shuffle(train_data)
            for train_ex in train_data:
                g, h = self.find_BMU(train_ex)
                self.SOM = self.update_weights(train_ex,
                                     learn_rate, radius_sq, (g, h))
            # Update learning rate and radius
            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)
        return self.SOM