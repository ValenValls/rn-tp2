import numpy as np
from matplotlib import pyplot as pl



class Hebbiano:


    def __init__(self, N,M):        
        self.weights = np.random.normal( 0, 0.1, (N,M))
        self.m = M
        self.n = N  

    #Para ver que tan cerca se esta de converger y decidir si parar, utilizamos la ortogonalidad de los pesos
    def ortogonalidad(self):
        return np.sum(np.abs(np.dot( self.weights.T, self.weights) - np.identity(self.m) ))/2

    #Entrenamiento del modelo con el algoritmo de Oja
    
    #Los pesos se van al infinito, en los ejercicios de la practica me dejo de pasar cuando normalice los datos
    #Puede que el error venga del lado normalizacion
    def trainOja(self,X,error_limit=0.01, limit=100):
        t = 1

        while t < limit and (self.ortogonalidad() > error_limit):
            #Con este lr adaptativo da overflow
            #learning_rate = 1 / t
            #Tal vez algo de este estilo?
            #learning_rate = 0.0001/t
            #Me hace ruido el overflow igual, despues reviso las cuentas
            learning_rate = 0.0001
            for x in X:
                #Las cuentas me coinciden con las que yo tenia
                #Pero tuve que hacer este reshape porque x es un vector
                x = x.reshape((1, -1))
                Y = np.dot( x, self.weights)
                Z = np.dot( Y, self.weights.T)
                dW = np.outer( x-Z, Y)
                self.weights+= learning_rate * dW
            t += 1
        #La ortogonalidad me da cerca de 2 cuando sale, asi que claramente hay algo mal
        #print(self.ortogonalidad())

    #Entrenamiento del modelo con el algoritmo de Sanger

    #Hay error con la resta de x.T y Z, me parece que hay q meter reshape en algun lado, HECHO
    def trainSanger(self,X,error_limit=0.01, limit=100):
        t = 1        
        while t<limit and (self.ortogonalidad() > error_limit):
            #learning_rate=1/t
            learning_rate = 0.0001/t
            #learning_rate = 0.0001
            for x in X:
                x = x.reshape((1, -1))
                Y = np.dot( x, self.weights)
                D = np.triu( np.ones((self.m,self.m)))
                Z = np.dot( self.weights, Y.T*D)
                dW = (x.T - Z) * Y
                self.weights+= learning_rate * dW  
            t += 1
        #La ortogonalidad de este dio cerca de 0.1 asi que esta bastante bien creo
        #print(self.ortogonalidad())


    #Prediccion con el modelo entrenado
    def predict(self, X):
        Y = np.dot( X, self.weights)
        return Y


