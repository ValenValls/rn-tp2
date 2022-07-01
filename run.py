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
        parser.add_argument('--data_file', '-df', type=str, default='./data/tp2_training_dataset.csv',
                            help='Archivo correspondiente a los Datos de entrada, por default el correspondiente al ejercicio 1')
        parser.add_argument('--model', '-m', choices=['som', 'hebb'], default='hebb',
                            help='Modelo a correr, por default Hebbiano con Oja')
        parser.add_argument('--save', '-s', const=True, nargs='?',
                            help='Guarda los datos del modelo entrenado', default=False)
        parser.add_argument('--out_modelo_file', '-omf', type=str, default='modelo_entrenado.txt',
                            help='Archivo de salida del modelo previamente entrenado')
        parser.add_argument('--out_data_file', '-odf', type=str, default='predicciones.txt',
                            help='Archivo de salida de las predicciones')
        parser.add_argument('--args', '-a', nargs='*', default=[850,9,'oja'])
        parser.parse_args(namespace=self)

if __name__ == '__main__':
    run = Consola()
    X, Y = get_data(run.data_file)
    X = normalize(X)
    run.args[0]=int(run.args[0])
    run.args[1]= int(run.args[1])
    print (*run.args)
    modelo = MODELO[run.model](*run.args)



    if run.modelo_file:
        # Si se indica un archivo de modelo por entrada, se asume que se quiere levantar ese modelo para predecir el data_file
        # todo definir el formato de archivo, etc
        assert True == False, 'Falta implementar esto'
    else:
         # Si no tengo un modelo tengo que entrenar de cero segun el ejercicio.
        if run.model == 'hebb':
            modelo.train(X)
            predicted = modelo.predict(X)
            plot3D(predicted, Y, run.args[2].upper()) # el tercer dato es el tipo de regla
        else:
            modelo.train(X)
            total_epochs = 0
            for epochs, i in zip([1, 4, 5, 10], range(0, 4)):
                total_epochs += epochs
                SOM = modelo.train(X, epochs=epochs)
            udm = u_matrix(SOM, run.args[1])
            plotSOM(udm, run.args[1]) # el segundo dato es M
            plt.show()

