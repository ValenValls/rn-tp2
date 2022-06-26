import numpy as np

 #Por lo que entendi, para este TP no es necesario separar el dataset en train y test, sino que se entrena con todo
 #Pero puede ser necesario permutar los datos para analizar si cambian los resultados
 
def get_data(archivo_data):
    #Los datos tienen la categoria en la primera columna, y 850 atributos más  
   
    X = np.genfromtxt(archivo_data,dtype='float',delimiter=',',usecols=range(1,851))
    Y = np.genfromtxt(archivo_data, dtype='str',delimiter=',', usecols=(0,))

    return X, Y



#Hay que normalizar los datos
def normalize(X):
    mean = X.mean(0)
    std = X.std(0) 
    X = (X - mean) / np.square(std)
    return X


