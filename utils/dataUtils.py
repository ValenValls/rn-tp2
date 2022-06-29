import numpy as np
from matplotlib import pyplot as plt
 #Por lo que entendi, para este TP no es necesario separar el dataset en train y test, sino que se entrena con todo
 #Pero puede ser necesario permutar los datos para analizar si cambian los resultados
 
def get_data(archivo_data):
    #Los datos tienen la categoria en la primera columna, y 850 atributos más  
   
    X = np.genfromtxt(archivo_data,dtype='float',delimiter=',',usecols=range(1,851))
    Y = np.genfromtxt(archivo_data, dtype='float',delimiter=',', usecols=(0,))

    return X, Y



#Hay que normalizar los datos para el aprendizaje hebbiano
def normalize(X):
    mean = X.mean(0)
    std = X.std(0) 
    X = (X - mean) / np.square(std)
    return X

def plot3D(X,Y):
    Y = Y.reshape((-1, 1))

    #Esto define distintos angulos desde el que se toman los graficos en R3
    #Capaz convengan otras inclinaciones
    proyecciones = [(60, 60), (150, 60),
                    (60, 150), (240, 240), ]
    for i in range(0,9,3):
        for j, (a, b) in enumerate(proyecciones):
            #plt.figure(figsize=(8, 8))
            ax = plt.subplot(2, 2, j + 1, projection="3d")
            ax.scatter3D(X[:, i], X[:, i+1], X[:, i+2], c=Y)
            ax.view_init(a, b)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        plt.suptitle('Componentes Y{},Y{},Y{}'.format(i+1,i+2,i+3))
        plt.show()


