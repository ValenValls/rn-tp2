import numpy as np
from matplotlib import pyplot as plt, cm


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
#Agrego title para diferenciar los graficos
def plot3D(X,Y,title):
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
        plt.suptitle(title + ': '+ 'Componentes Y{},Y{},Y{}'.format(i+1,i+2,i+3))
        plt.show()

#Estos tres metodos los estoy dejando por ahora solo para probar que anda el RUN correctamente. Posiblemente no sirva para nada
#TODO en tal caso borrarlo luego
def plotSOM(udm, m):
    fig = plt.figure(figsize=(4, 4), dpi=120)
    axes1 = fig.add_subplot(111, projection="3d")
    ax = fig.gca()
    xx_1 = np.arange(1, m - 1, 1)
    xx_2 = np.arange(1, m - 1, 1)
    x_1, x_2 = np.meshgrid(xx_1, xx_2)
    Z = np.array([[udm[i][j] for i in range(m - 2)] for j in range(m - 2)])
    ax.set_zlim(0, .5)
    ax.plot_surface(x_1, x_2, Z, cmap=cm.gray)
    plt.xlabel('$i$', fontsize=11)
    plt.ylabel('$j$', fontsize=11)
    plt.title("U-matrix", fontsize=11)
# U-matrix
def dist3(p1,p2):
    """
    Square of the Euclidean distance between points p1 and p2
    in 3 dimensions.
    """
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2

def u_matrix(SOM, m):
    udm = np.zeros((m - 2, m - 2))  # initiaize U-matrix with elements set to 0
    for i in range(1, m - 1):  # loops over the neurons in the grid
        for j in range(1, m - 1):
            udm[i - 1][j - 1] = np.sqrt(dist3(SOM[i][j], SOM[i][j + 1]) + dist3(SOM[i][j], SOM[i][j - 1]) +
                                        dist3(SOM[i][j], SOM[i + 1][j]) + dist3(SOM[i][j], SOM[i - 1][j]))
    return udm