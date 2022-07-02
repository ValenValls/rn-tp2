import argparse
from hebbiano.model import Hebbiano
from SOM.model import SOM
from utils.dataUtils import *
import numpy as np
from matplotlib import pyplot as plt

MODELO = {'hebb': Hebbiano,
          'som': SOM}


class Consola:

    def __init__(self):
        self.parse_args()

    def parse_args(self):
        parser = argparse.ArgumentParser(prog='rn-tp1')
        parser.add_argument('--modelo_file', '-mf', type=str, default=None,
                            help='Archivo de entrada del modelo previamente entrenado')
        parser.add_argument('--trans_file', '-tf', type=str, default=None,
                            help='Archivo de entrada con las transformaciones aplicadas al modelo previamente entrenado')
        parser.add_argument('--data_file', '-df', type=str, default='./data/tp2_training_dataset.csv',
                            help='Archivo correspondiente a los Datos de entrada, por default el del tp')
        parser.add_argument('--model', '-m', choices=['som', 'hebb'], default='hebb',
                            help='Modelo a correr, por default Hebbiano con Oja')
        parser.add_argument('--save', '-s', const=True, nargs='?',
                            help='Guarda los datos del modelo entrenado', default=False)
        parser.add_argument('--out_modelo_file', '-omf', type=str, default='modelo_entrenado.txt',
                            help='Archivo de salida del modelo previamente entrenado')
        parser.add_argument('--out_trans_file', '-otf', type=str, default='transformaciones.txt',
                            help='Archivo de salida del modelo previamente entrenado')
        parser.add_argument('--out_data_file', '-odf', type=str, default='predicciones.txt',
                            help='Archivo de salida de las predicciones')
        parser.add_argument('--args', '-a', nargs='*', default=None)
        parser.parse_args(namespace=self)

if __name__ == '__main__':
    run = Consola()
    X, Y = get_data(run.data_file)

    #TODO dependiendo de si guardemos medias y std se normaliza aca o en cada opcion por separado
    X, mean, std = normalize(X)

    if run.args:
        # Si recibo argumentos los dos primeros siempre son N y M. Los recibo como strings y debo convertirlos en numeros
        #entrada
        run.args[0]=int(run.args[0])
        M = run.args[0]

        #salida
        run.args[1]= int(run.args[1])
        N=run.args[1]

        print (*run.args)
    #Si no recibo argumentos se generara un modelo vacio, que se completara cuando importe el modelo.
    #Si voy a entrenar de cero si o si debe recibir argumentos.
    modelo = MODELO[run.model](*run.args) if run.args else MODELO[run.model]()

    regla_modelo = ''



    if run.modelo_file:
        # Si se indica un archivo de modelo por entrada, se asume que se quiere levantar ese modelo para predecir el data_file
        # todo definir el formato de archivo, etc
        modelo.import_model(run.modelo_file)
        if run.model == 'hebb':
            predicted = modelo.predict(X)
            regla_modelo = modelo.reglas + '_'
            plot3D(predicted, Y, regla_modelo) # el tercer dato es el tipo de regla
        else:
            #TODO hay que terminar el som para que esto pueda hacer algo
            plotSOM(modelo.u_matrix(), modelo.m)  # el segundo dato es M
            plt.show()
            assert True == False, 'Falta implementar esto'
    else:
         # Si no tengo un modelo tengo que entrenar de cero segun el ejercicio.


        if run.model == 'hebb':
            modelo.train(X)
            predicted = modelo.predict(X)
            regla_modelo = run.args[2].upper() + '_'
            plot3D(predicted, Y, regla_modelo) # el tercer dato es el tipo de regla
        else:
            modelo.train(X)
            # TODO esto era para ir graficando valores intermedios. Creo que ya no es necesario.
            # total_epochs = 0
            # for epochs, i in zip([1, 4, 5, 10], range(0, 4)):
            #     total_epochs += epochs
            #     SOM = modelo.train(X, epochs=epochs)
            ##TODO me parece que hice algo mal al pasar el grafico porque sale raro
            # igual tampoco se que estamos graficando
            # udm = u_matrix(SOM, run.args[1])
            plotSOM(modelo.u_matrix(), modelo.m) # el segundo dato es M
            plt.show()

        if run.save:
             # TODO esto era para guardar medias y std para normalizar nuevos valores
             #  pero en ese caso tambien habria que readaptar la funcion para que pueda
             #  normalizar a partir de ellos. DEFINIR
             # with open(f"{run.model}_{regla_modelo}{run.out_trans_file}", 'w') as f:
             #     f.write(f"{N}\n")
             #     f.write(f"{','.join([str(n) for n in mean])}\n")
             #     f.write(f"{','.join([str(n) for n in std])}\n")
             modelo.export_model(f"{run.model}_{regla_modelo}_{run.out_modelo_file}")