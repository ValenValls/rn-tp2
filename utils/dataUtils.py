import numpy as np
from matplotlib import pyplot as plt, cm


#Por lo que entendi, para este TP no es necesario separar el dataset en train y test, sino que se entrena con todo
 #Pero puede ser necesario permutar los datos para analizar si cambian los resultados
 
def get_data(archivo_data):
    #Los datos tienen la categoria en la primera columna, y 850 atributos más  
   
    X = np.genfromtxt(archivo_data,dtype='float',delimiter=',',usecols=range(1,851))
    Y = np.genfromtxt(archivo_data, dtype='int',delimiter=',', usecols=(0,))

    return X, Y

#Viendo la parte del enunciado de mapeo de caracteristicas, ahí habla de ver diferencia de datos de entrenamiendo y validacion
# Si se le da X,Y obtenidos de get_data, separa aleatoriamente en xtrain,ytrain y xvalidation,yvalidation
#   Dejo la version de separacion aleatoria que puede perder la uniformidad que tenía el dataset
#   Podría llegar a servir para ver la diferencia cuando se entrena con y sin uniformidad de datos?
def random_separate_train_validation(X,Y,validation_size=0.1, total_regs=0):
    total_regs = X.shape[0] if total_regs == 0 else total_regs
    X = X[:total_regs, : ]
    Y = Y[:total_regs]
    #Tomo el porcentaje de datos de entrenamiento
    end_train = total_regs - int(validation_size * total_regs)
    #Tomo los datos de entrenamiento de manera aleatoria
    indexes = np.random.permutation(total_regs)
    trn = indexes[:end_train]  
    val = indexes[end_train:]
    #Separo
    X_train = X[trn, :]
    Y_train = Y[trn]
    X_validation = X[val, :]
    Y_validation = Y[val]
    return X_train, Y_train, X_validation, Y_validation

#Haciendo esto note que no hay exactamente 100 instancias de cada categoria en el dataset, esta separacion igualmente...
#   mantiene misma proporcion que el dataset original
def proportional_separate_train_validation(X,Y,validation_size=0.1, total_regs=0):
    total_regs = X.shape[0] if total_regs == 0 else total_regs
    X = X[:total_regs, : ]
    Y = Y[:total_regs]  
    #Separo los indices por categoria    
    indexes_by_cat = [[],[],[],[],[],[],[],[],[]] 
    for idy in range(0,len(Y)):
        indexes_by_cat[Y[idy]-1].append(idy)           
    #Separo los indices entre entrenamiento y validacion       
    train_by_cat = []
    validate_by_cat = []
    for c in range(1,10):       
            category_regs = len(indexes_by_cat[c-1])            
            end_train = category_regs - int(validation_size * category_regs)            
            np.random.shuffle(indexes_by_cat[c-1])
            train_by_cat += indexes_by_cat[c-1][:end_train]
            validate_by_cat += indexes_by_cat[c-1][end_train:]        
    #Separo el data set con los indices
    X_train = X[train_by_cat, :]
    Y_train = Y[train_by_cat]
    X_validation = X[validate_by_cat, :]
    Y_validation = Y[validate_by_cat]
    return X_train, Y_train, X_validation, Y_validation

#Recibe el dataset a separar, Y la categorias
#Se puede utilizar con X_validation, Y_validation luego de
#Mantener el Y no es necesario, ya que estan separados
def separate_attributes_by_category(X,Y):
    indexes_by_cat = [[],[],[],[],[],[],[],[],[]]
    for idy in range(0,len(Y)):
        indexes_by_cat[Y[idy]-1].append(idy) 
    Xs_by_category = []
    for c in range(1,10):    
        Xs_by_category.append(X[indexes_by_cat[c-1], :])
    return Xs_by_category
        

#Hay que normalizar los datos para el aprendizaje hebbiano
#Devuelvo mean y std para exportarlos al guardar el modelo
def normalize(X):
    mean = X.mean(0)
    std = X.std(0) 
    X = (X - mean) / np.square(std)
    return X, mean, std
#Agrego title para diferenciar los graficos
def plot3D(X,Y,title):
    Y = Y.reshape((-1, 1))

    #Esto define distintos angulos desde el que se toman los graficos en R3
    #Capaz convengan otras inclinaciones
    proyecciones = [(60, 60), (150, 60),
                    (60, 150), (240, 240), ]

    plt.figure(figsize=(12, 10))


    for i in range(0,9,3):

        for j, (a, b) in enumerate(proyecciones):
            ax = plt.subplot(3, 4, int(4*i/3)+j + 1, projection="3d")
            ax.scatter3D(X[:, i], X[:, i+1], X[:, i+2], c=Y,cmap='Set1',marker='.', lw=0)
            ax.view_init(a, b)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        plt.subplot(3, 4, int(4*i/3) + 2).set_title('Componentes Principales Y{},Y{},Y{}'.format(i+1, i + 2, i + 3), x=1.2, y=1.1, fontsize=12)
    plt.suptitle(title, fontsize=14)
    plt.subplots_adjust(hspace=.2, wspace=.01, top=.9)
    plt.show()









def plotSOMColorMap(categorieMatrix):
    plt.imshow(categorieMatrix, cmap='Pastel1')


    for y in range(categorieMatrix.shape[0]):
        for x in range(categorieMatrix.shape[1]):
            plt.text(x , y, '%i' % categorieMatrix[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     )

    plt.suptitle('Categoría más popular por Unidad de Salida')
    plt.show()

#Estos tres metodos los estoy dejando por ahora solo para probar que anda el RUN correctamente. Posiblemente no sirva para nada
# #TODO en tal caso borrarlo luego
# def plotSOM(udm, m):
#     fig = plt.figure(figsize=(4, 4), dpi=120)
#     axes1 = fig.add_subplot(111, projection="3d")
#     ax = fig.gca()
#     xx_1 = np.arange(1, m - 1, 1)
#     xx_2 = np.arange(1, m - 1, 1)
#     x_1, x_2 = np.meshgrid(xx_1, xx_2)
#     Z = np.array([[udm[i][j] for i in range(m - 2)] for j in range(m - 2)])
#     ax.set_zlim(0, .5)
#     ax.plot_surface(x_1, x_2, Z, cmap=cm.gray)
#     plt.xlabel('$i$', fontsize=11)
#     plt.ylabel('$j$', fontsize=11)
#     plt.title("U-matrix", fontsize=11)
# U-matrix
# def dist3(p1,p2):
#     """
#     Square of the Euclidean distance between points p1 and p2
#     in 3 dimensions.
#     """
#     return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2
#
# def u_matrix(SOM, m):
#     udm = np.zeros((m - 2, m - 2))  # initiaize U-matrix with elements set to 0
#     for i in range(1, m - 1):  # loops over the neurons in the grid
#         for j in range(1, m - 1):
#             udm[i - 1][j - 1] = np.sqrt(dist3(SOM[i][j], SOM[i][j + 1]) + dist3(SOM[i][j], SOM[i][j - 1]) +
#                                         dist3(SOM[i][j], SOM[i + 1][j]) + dist3(SOM[i][j], SOM[i - 1][j]))
#     return udm