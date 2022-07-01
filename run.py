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

            # print(predictedOja)
            # print(predictedSanger)


    #     X_train, X_test, Y_train, Y_test = random_separate_train_test(X, Y, test_size=0.1)
    #     print('Normalizando los datos de entrada...')
    #     X_train, mean, std = normalize(X_train)
    #     X_test, _, _ = normalize(X_test, mean=mean, std=std)
    #     if run.save:
    #         with open(f"{run.ejercicio}_{run.out_trans_file}", 'w') as ft:
    #             ft.write(f"{','.join([str(n) for n in mean])}\n")
    #             ft.write(f"{','.join([str(n) for n in std])}\n")
    #     print('Transformando la salida')
    #     if run.ejercicio == 1:
    #         Y_train = labelizeMB(Y_train)
    #         Y_test = labelizeMB(Y_test)
    #     else:
    #         Y_train = np.log(Y_train)
    #         Y_test = np.log(Y_test)
    #
    #     # probamos varias combinaciones de hyperparametros, vamos guardando la que mejor validacion tenga.
    #     # al final entrenamos con TODOS los datos un unico modelo y lo guardamos.
    #     mejor_modelo = [0, 0]
    #     mejor_validacion = np.infty
    #
    #
    #
    #
    #     if run.ejercicio == 1:
    #         perceptron_params = [{'layers':[10,20,2,1],
    #                              'activations':['sigmoid', 'sigmoid', 'escalon']},
    #                             {'layers': [10, 30, 1],
    #                              'activations': ['relu', 'sigmoid']},
    #                             {'layers': [10, 30, 15, 1],
    #                              'activations': ['relu', 'relu', 'sigmoid']},
    #                             ]
    #         train_params = [{'learning_rate':0.05,
    #                         'error_limit':0.01,
    #                         'epoch_limit':10000,
    #                          'batches': 100,
    #                         'up_limit':10000}, #modelo sin early stopping
    #                         {'learning_rate':0.06,
    #                          'error_limit':0.01,
    #                          'epoch_limit':10000,
    #                          'batches': 100,
    #                          'up_limit':2000},
    #                         {'learning_rate': 0.06,
    #                          'error_limit': 0.01,
    #                          'epoch_limit': 10000,
    #                          'batches': 0,
    #                          'up_limit': 10000} # modelo sin batches y sin early stopping ,
    #                         ]
    #     else:
    #         # HYPERPARAMETER SEARCH EJERCICIO 2
    #         #todo solo dejo sin comentar el optimo, avisen si cambia porque esta en el informe
    #         perceptron_params = [{'layers':[8, 32, 24, 16, 8, 4, 2],
    #                               'activations':['relu','relu','relu','relu','relu','relu']},
    #                             {'layers': [8, 64, 64, 2],
    #                               'activations': ['relu', 'relu', 'identidad']}
    #                              ]
    #         train_params = [{'learning_rate':0.01,
    #                          'error_limit':0.01,
    #                          'epoch_limit':1000,
    #                          'up_limit':50,
    #                          'batches': 30,
    #                          'cost': 'mse'},
    #                         {'learning_rate': 0.01,
    #                          'error_limit': 0.01,
    #                          'epoch_limit': 5000,
    #                          'batches': 50,
    #                          'up_limit': 200,
    #                          'cost': 'mse'}
    #                         ]
    #
    #     for p_idx, pct_param in enumerate(perceptron_params):
    #         for t_idx, trn_param in enumerate(train_params):
    #                 print(f"Entrenando modelo {p_idx} - {t_idx}")
    #                 perceptron = Perceptron(**pct_param)
    #                 train_error, val_error = perceptron.train_model_while_validating(X_train, Y_train, X_test,
    #                                                                                   Y_test,**trn_param)
    #
    #                 plt.plot(train_error, 'b',marker='o',markersize=2, label='Train error')
    #                 plt.plot(val_error, 'r',marker='o', markersize=2, label='Validation error')
    #                 plt.title(f"Ejercicio {run.ejercicio} error modelo: {p_idx} - {t_idx}")
    #                 plt.legend()
    #
    #                 if run.save:
    #                     plt.savefig(f"ej_{run.ejercicio}_{p_idx}_{t_idx}_error_modelo.png")
    #                     plt.show()
    #                     plt.close()
    #
    #                 Y_train_predicted = perceptron.predict(X_train)
    #                 Y_test_predicted = perceptron.predict(X_test)
    #
    #                 if run.ejercicio == 1:
    #                     # En ambos casos binarizo la salida
    #                     Y_train_predicted = np.where(Y_train_predicted <= 0.5, 0, 1)
    #                     Y_test_predicted = np.where(Y_test_predicted <= 0.5, 0, 1)
    #
    #                 plt.plot(Y_train, 'b', label='Y train')
    #                 plt.plot(Y_test, 'r', label='Y test')
    #                 plt.plot(Y_train_predicted.T, 'y', label='Y train predicted')
    #                 plt.plot(Y_test_predicted.T, 'g', label='Y test predicted')
    #                 plt.title(f"Ejercicio {run.ejercicio} error modelo: {p_idx} - {t_idx}")
    #                 plt.legend()
    #
    #                 if run.save:
    #                     plt.savefig(f"ej_{run.ejercicio}_{p_idx}_{t_idx}_resultados.png")
    #                     plt.show()
    #                     plt.close()
    #
    #
    #                 if run.ejercicio == 1:
    #                     Y_train_error = np.sum(abs(Y_train_predicted - Y_train.T)) / Y_train.shape[0]
    #                     Y_test_error = np.sum(abs(Y_test_predicted - Y_test.T)) / Y_test.shape[0]
    #                     print(f"Se predijo correctamente {1 - Y_train_error} del train set.")
    #                     print(f"Se predijo correctamente {1 - Y_test_error} del test set.")
    #
    #                 else:
    #                     Y_train_error = perceptron.calcular_error(Y_train_predicted, Y_train.T, 'mse')
    #                     Y_test_error = perceptron.calcular_error(Y_test_predicted, Y_test.T, 'mse')
    #                     print(f"El MSE del train set es {Y_train_error}")
    #                     print(f"El MSE del test set es {Y_test_error}")
    #
    #                 if Y_test_error < mejor_validacion:
    #                     mejor_modelo = [p_idx, t_idx]
    #                     mejor_validacion = Y_test_error
    #
    #                 if run.save:
    #                     perceptron.export_model(f"ej_{run.ejercicio}_{p_idx}_{t_idx}_{run.out_modelo_file}")
    #                     with open(f'ej_{run.ejercicio}_testeados.txt', 'a+', encoding='utf-8') as results:
    #                         results.write(f"EJERCICIO {run.ejercicio} MODELO {p_idx} - {t_idx}\n")
    #                         results.write(str(pct_param))
    #                         results.write('\n')
    #                         results.write(str(trn_param))
    #                         results.write('\n')
    #                         results.write(f"El error obtenido para el train set es de {Y_train_error}\n")
    #                         results.write(f"El error obtenido para el test set es de {Y_test_error}\n")
    #
    #     print("Entrenando el mejor modelo con TODA la entrada")
    #     mejor_perceptron = Perceptron(**perceptron_params[mejor_modelo[0]])
    #     mejor_params = train_params[mejor_modelo[1]]
    #     mejor_params['up_limit'] = mejor_params['epoch_limit']
    #     X, mean, std = normalize(X)
    #     if run.ejercicio == 1:
    #         Y = labelizeMB(Y)
    #     else:
    #         Y = np.log(Y)
    #
    #     if run.save:
    #         with open(f"{run.ejercicio}_MEJOR_{run.out_trans_file}", 'w') as ft:
    #             ft.write(f"{','.join([str(n) for n in mean])}\n")
    #             ft.write(f"{','.join([str(n) for n in std])}\n")
    #     train_error, _ = mejor_perceptron.train_model_while_validating(X, Y, **train_params[mejor_modelo[1]])
    #     Y_predicted = mejor_perceptron.predict(X)
    #     if run.ejercicio ==1:
    #         Y_predicted = np.where(Y_predicted <= 0.5, 0, 1)
    #         Y_error =  np.sum(abs(Y_predicted - Y.T)) / Y.shape[0]
    #         print(f"Se predijo correctamente {1 - Y_error} del full train set.")
    #
    #     if run.save:
    #         mejor_perceptron.export_model(f"{run.ejercicio}_MEJOR_{run.out_modelo_file}")
    #
    #
    #
    #
    #
    #
    #
