import numpy as np
from matplotlib import pyplot as plt, cm



def get_data(archivo_data):
    #Los datos tienen la categoria en la primera columna, y 850 atributos más  
   
    X = np.genfromtxt(archivo_data,dtype='float',delimiter=',',usecols=range(1,851))
    Y = np.genfromtxt(archivo_data, dtype='int',delimiter=',', usecols=(0,))

    return X, Y

# Si se le da X,Y obtenidos de get_data, separa aleatoriamente en xtrain, ytrain y xvalidation, yvalidation
#   No mantiene uniformidad de categorías
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

#Si se le da X,Y obtenidos de get_data, separa aleatoriamente en xtrain, ytrain y xvalidation, yvalidation
#   Mantiene uniformidad de categorías
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

#Recibe X el dataset a separar segun las Y categorias, devuelve dataset separado
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

