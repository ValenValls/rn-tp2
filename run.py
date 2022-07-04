import argparse
from hebbiano.model import Hebbiano
from SOM.model import SOM
from utils.dataUtils import *
import numpy as np

import os

MODELO = {'hebb': Hebbiano,
          'som': SOM}


class Consola:

    def __init__(self):
        self.parse_args()

    def parse_args(self):
        parser = argparse.ArgumentParser(prog='rn-tp2')
        parser.add_argument('--modelo_file', '-mf', type=str, default=None,
                            help='Archivo de entrada del modelo previamente entrenado')
        parser.add_argument('--data_file', '-df', type=str, default='./data/tp2_training_dataset.csv',
                            help='Archivo correspondiente a los Datos de entrada, por default el del tp')
        parser.add_argument('--model', '-m', choices=['som', 'hebb'], default='hebb',
                            help='Modelo a correr, por default Hebbiano con Oja')
        parser.add_argument('--save', '-s', const=True, nargs='?',
                            help='Guarda los datos del modelo entrenado', default=False)
        parser.add_argument('--graph', '-g', const=True, nargs='?',
                            help='Grafica las salidas del SOM', default=False)
        parser.add_argument('--out_modelo_file', '-omf', type=str, default='modelo_entrenado.txt',
                            help='Archivo de salida del modelo previamente entrenado')
        parser.add_argument('--out_data_file', '-odf', type=str, default='predicciones.txt',
                            help='Archivo de salida de las predicciones')
        parser.add_argument('--args', '-a', nargs='*', default=None)
        parser.parse_args(namespace=self)

if __name__ == '__main__':
    run = Consola()
    X, Y = get_data(run.data_file)
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
        modelo.import_model(run.modelo_file, run.graph)
        if run.model == 'hebb':

            predicted = modelo.predict(X)
            regla_modelo = modelo.reglas + '_' #Al crear el modelo me guarda la regla en modelo_reglas
            plot3D(predicted, Y, regla_modelo)
        else:
            #TODO hay que terminar el som para que esto pueda hacer algo
            if run.graph:
                modelo.categorize_and_map(X, Y, title='Categorizacion sobre modelo pre-entrenado', fn='salida_preentrenado.png')
            predicted = modelo.categorize(X)
            if run.save:
                with open(run.out_data_file, 'w', encoding='utf-8') as f:
                    np.savetxt(f,predicted)
            acc_val, tot, ok = modelo.accuracy(Y, predicted)
            print(f'Validation accuracy: {acc_val}')
            for i, (t, e) in enumerate(zip(tot, ok)):
                if t > 0:
                    print(f'categoria {i}: {e / t}')
            # assert True == False, 'Falta implementar esto'
    else:
         # Si no tengo un modelo tengo que entrenar de cero segun el ejercicio.


        if run.model == 'hebb':

            modelo.train(X)
            predicted = modelo.predict(X)
            regla_modelo = run.args[2].upper() + '_'
            plot3D(predicted, Y, regla_modelo) # el tercer dato es el tipo de regla
        else:
            header = 'learning_rate,radio,m,validation,epochs,training_acc,validation_acc,sin_cat,cat_1,cat_2,cat_3,cat_4,cat_5,cat_6,cat_7,cat_8,cat_9\n'
            with open('results.csv', '+w', encoding='utf-8') as file_acc:
                file_acc.write(header)

                hyper_params = [{'lr': 0.01, 'r': 1, 'm': 3, 'val':0.1, 'e':8},
                                {'lr': 0.01, 'r': 1, 'm': 9, 'val':0.1, 'e':16},
                                {'lr': 0.01, 'r': 3, 'm': 3, 'val':0.1, 'e':16},
                                {'lr': 1, 'r': 10, 'm': 9, 'val':0.1, 'e':16},
                                {'lr': 0.01, 'r': 1, 'm': 8, 'val': 0.2, 'e': 40},
                                {'lr': 0.3, 'r': 3, 'm': 6, 'val':0.2, 'e':24},
                                {'lr': 0.75, 'r': 3, 'm': 8, 'val':0.2, 'e':40}]

                best_accuracy = 0
                best_model = None

                for hp in hyper_params:
                    val = hp['val']
                    lr = hp['lr']
                    r = hp['r']
                    m = hp['m']
                    e = hp['e']

                    path = f'SOM_lr_{lr}_r_{r}_m_{m}_val_{val}_e_{e}'
                    if run.graph and not os.path.exists(path):
                        os.mkdir(path)

                    print(f'Calculando modelo con learning rate: {lr} - influence radius:{r} - m: {m} - validation size: {val} - epochs: {e}')

                    X_train, Y_train, X_val, Y_val = proportional_separate_train_validation(X, Y,validation_size=val)
                    modelo.change_m_and_reset(m)
                    SOM, acc = modelo.train(X_train, learn_rate=lr, radius_sq=r, epochs=e, graph=run.graph, Y=Y_train, path= path, fn='SOM')

                    if run.graph:
                        title = f"Clasificación del set de validación learning rate:{lr} influence radius:{r} m:{m} validation size:{val} epochs:{e}"
                        modelo.categorize_and_map(X_val, Y_val, title=title, path=path, fn='SOM_val_')
                        title = f"Clasificación del training set learning rate:{lr} influence radius:{r} m:{m} validation size:{val} epochs:{e}"
                        modelo.categorize_and_map(X_train, Y_train, title=title, path=path, fn='SOM_train_')

                    print (f'Training accuracy= {acc}')

                    predicted = modelo.categorize(X_val)
                    acc_val, tot, ok = modelo.accuracy(Y_val, predicted)
                    print(f'Validation accuracy: {acc_val}')

                    #Guardo el modelo que mayor accuracy tiene sobre el conjunto de validacion.
                    if acc_val > best_accuracy:
                        best_accuracy= acc_val
                        best_model = hp

                    res = [f'{lr:.6f}', str(r), str(m), f'{val:.6f}', str(e)]
                    res.append(f'{acc:.6f}')
                    res.append(f'{acc_val:.6f}')

                    acc_cat = np.zeros(10)
                    for i, (t,e ) in enumerate(zip(tot, ok)):
                        if t > 0:
                            res.append(f'{e/t:.6f}')
                            print(f'categoria {i}: {e/t}')
                        else:
                            res.append('-')
                    file_acc.write(','.join(res) + '\n')

                # DEJO EL MEJOR MODELO entrenado con el set completo de datos:
                val = best_model['val']
                lr = best_model['lr']
                r = best_model['r']
                m = best_model['m']
                e = best_model['e']
                path = f'SOM_BEST_TRAINED'
                if not os.path.exists(path):
                    os.mkdir(path)
                modelo.change_m_and_reset(m)
                SOM, acc = modelo.train(X, learn_rate=lr, radius_sq=r, epochs=e, graph=True, Y=Y, path=path,
                                        fn='BEST_SOM')
                title = f"Clasificación del set de validación learning rate:{lr} influence radius:{r} m:{m} validation size:{val} epochs:{e}"
                modelo.categorize_and_map(X_val, Y_val, title=title, path=path, fn='BEST_SOM_val_')
                title = f"Clasificación del training set learning rate:{lr} influence radius:{r} m:{m} validation size:{val} epochs:{e}"
                modelo.categorize_and_map(X_train, Y_train, title=title, path=path, fn='BEST_SOM_train_')

        if run.save:
             modelo.export_model(f"{run.model}_{regla_modelo}_{run.out_modelo_file}", run.graph)