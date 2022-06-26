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
            learning_rate = 1/t
            for x in X:
                Y = np.dot( x, self.weights)
                Z = np.dot( Y, self.weights.T)
                dW = np.outer( x-Z, Y)                
                self.weights+= learning_rate * dW
            t += 1

    #Entrenamiento del modelo con el algoritmo de Sanger

    #Hay error con la resta de x.T y Z, me parece que hay q meter reshape en algun lado 
    def trainSanger(self,X,error_limit=0.01, limit=5):
        t = 1        
        while t<limit and (self.ortogonalidad() > error_limit):
            learning_rate = 1/t
            for x in X:
                
                Y = np.dot( x, self.weights)
                D = np.triu( np.ones((self.m,self.m)))
                Z = np.dot( self.weights, Y.T*D)
                dW = (x.T - Z) * Y
                self.weights+= learning_rate * dW  
            t += 1 
                       



    #Prediccion con el modelo entrenado
    def predict(self, X):
        Y = np.dot( X, self.weights)
        return Y


