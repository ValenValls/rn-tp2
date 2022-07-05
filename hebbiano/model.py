import numpy as np
from matplotlib import pyplot as pl



class Hebbiano:


    def __init__(self, N=None,M=None, reglas=None):
        #if N and M :
            #Mejor
            #self.weights = np.random.normal( -0.1, 0.1, (N,M))
        #self.weights = np.random.uniform(-0.1, 0.1, (N, M))
        self.weights = np.random.normal(0, 0.0001, (N, M))
        #self.weights = np.random.randn(N, M) * 0.0001
        #self.weights = np.random.normal(0, 0.001, (N, M))
        #self.weights =np.random.normal(scale=0.25, size=(N, M))
        #todo no se si aporta en algo pero dejo por aca. Lei en general que los pesos se inicializaban con zero...
        # esto no funciona si lo inicializo con cero. Pero podriamos con numeros muy chicos como hicimos para
        # el perceptron self.weights = np.random.randn(N,M) * 0.0001 No se si cambia significativamente.
        self.m = M
        self.n = N
        self.reglas = reglas

    def import_model(self, filename, graph=False):
        # El formato del archivo seria
        # N = Cantidad de nodos entrada
        # M = Cantidad de nodos salida
        # string indicando 'oja' o 'sanger'
        # Matriz de pesos escrita como N filas de N elementos separados por coma
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"{self.n}\n")
            file.write(f"{self.m}\n")
            file.write(f"{self.reglas}\n")
            np.savetxt(file, self.weights, fmt='%.6f')

    def export_model(self, filename, graph=False):
        with open(filename, 'r', encoding='utf-8') as file:
            file_rows = file.readlines()
        self.n = int(file_rows[0])
        self.m = int(file_rows[1])
        self.reglas = file_rows[2]
        W = np.matrix((';').join([row[:-1] for row in file_rows[3: (3+self.n)]]))
        self.weights = np.asarray(W)
        assert self.weights.shape == (self.n, self.m)


    #Para ver que tan cerca se esta de converger y decidir si parar, utilizamos la ortogonalidad de los pesos
    def ortogonalidad(self):
        return np.sum(np.abs(np.dot( self.weights.T, self.weights) - np.identity(self.m) ))/2

    #Entrenamiento del modelo con el algoritmo de Oja
    
    def trainOja(self,X,error_limit=0.01, limit=100, lr=0.0001):
        t = 1
        X_train = X.copy()
        while t < limit and (self.ortogonalidad() > error_limit):
            # todo Poniendo pesos bajitos y limite 100 no corta por el error sino por el limite.
            #  Subiendo el limite, corta a las 250 vueltas aprox por el error.
            #  En una tercera prueba, bajo el limite de error a 0.001 corta a los 1000 con 0.002460829456370829 de
            #  error. Honestamente en ningun caso veo cambios notorios en los graficos.
            #  Aumentando el limite a 5000 corta con error 0.0009998011326659293
            #  a las 2453 iteraciones. Aumentando el learning rate da nans... esto se me hace raro puesto

            learning_rate = lr/t
            np.random.shuffle(X_train)
            for x in X_train:
                x = x.reshape((1, -1))
                Y = np.dot( x, self.weights)
                Z = np.dot( Y, self.weights.T)
                dW = np.outer( x-Z, Y)
                self.weights += learning_rate * dW
            t += 1

    #Entrenamiento del modelo con el algoritmo de Sanger


    def trainSanger(self,X,error_limit=0.01, limit=100, lr=0.0001):
        t = 1
        X_train = X.copy()
        while t<limit and (self.ortogonalidad() > error_limit):
            learning_rate = lr/t
            np.random.shuffle(X_train)
            for x in X_train:
                x = x.reshape((1, -1))
                Y = np.dot( x, self.weights)
                D = np.triu( np.ones((self.m,self.m)))
                Z = np.dot( self.weights, Y.T*D)
                dW = (x.T - Z) * Y
                self.weights += learning_rate * dW
            t += 1

    def train(self, X, error_limit=0.01, limit=500):
        if self.reglas == 'oja':
            self.trainOja(X, error_limit, limit)
        else:
            self.trainSanger(X, error_limit, limit)


    #Prediccion con el modelo entrenado
    def predict(self, X):
        Y = np.dot( X, self.weights)
        return Y


