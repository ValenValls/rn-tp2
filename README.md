# TRABAJO PRACTICO N.2 - REDES NEURONALES
Este proyecto cuenta con la implementacion de tres modelos de redes neuronales no supervisadas:

1. Modelo Hebbiano con regla de OJA
2. Modelo Hebbiano con regla de Sanger
3. Modelo de Mapeo de Caracteristicas (SOM) 


## Instalacion

Ubicarse en la carpeta del proyecto y correr por consola:

~~~bash
~$ python -m venv venv
~$ source ./venv/bin/activate
~$ pip install -r requirements.txt
~~~


## Entrenar Modelos

Para entrenar un modelo se debe correr por consola:

~~~bash
~$ python -m run [PARAMETROS DEL MODELO]
~~~

Los parametros que recibe son:

> **'--model', '-m'**
>
> Indica el modelo de base a correr: que puede ser `hebb` o `som`. Por defecto usa `hebb`
>
> **'--data_file', '-df'**
>
> Corresponde al archivo con los datos de entrada, se debe poner el path completo desde la carpeta del proyecto.
Por defecto este parametro es: `'./data/tp2_training_dataset.csv'`
>
> **'--save', '-s'**
>
> Por defecto `False`, guarda el modelo al finalizar el entrenamiento o los resultados predichos si se corre un modelo preentrenado para predecir.
>
> **'--graph', '-g'**
>
> Por defecto `False`, guarda graficos de los modelos entrenados para el SOM..
>
> **'--out_modelo_file', '-omf'**
>
> Por defecto `modelo_entrenado.txt`, indica el archivo de salida para guardar el modelo entrenado
> se le agrega como prefijo el tipo de modelo.
>
> **--modelo_file', '-mf'**
>
> Si recibe este parametro, en lugar de entrenar va a levantar el modelo previamente
> guardado en este archivo y predecir los datos de entrada. En este caso, no va a guardar el modelo sino que guardara
> los valores predecidos. Es importante indicar en el parametro `--model` el tipo de modelo al que corresponde.
>
> **'--out_data_file', '-odf'** 
>
> Corresponde al archivo de salida para los valores categorizados, por defecto `predicciones.txt`
>
> **'--args', '-a'** 
>
> Lista separada por espacios de los argumentos que debe recibir el modelo que corresponden a
> valor de M, valor de N y en el caso de hebb el tipo de regla 'oja' o 'sanjer'
>

EJEMPLOS:

~~~bash
~$ python -m run --model som -s --args 850 6
~$ python -m run --model som --modelo_file som__modelo_entrenado.txt
~$ python -m run --model hebb -s --args 850 9 oja
~$ python -m run --model hebb -s --args 850 9 sanger
~~~

## Salida

### HEBB
Si se corre con `-s` se guarda ademas un archivo con el mejor modelo obtenido. El formato de este archivo consiste en:

- Una primera linea con la cantidad de palabras N 
- Una segunda linea con un entero M que indica la cantidad de nodos de salida.
- Un string indicando el tipo de regla utilizada 'oja' o 'sanger'
- Las lineas siguientes corresponden a la matriz de pesos como N filas de M elementos

### SOM
Al correr SOM por defecto se guarda un archivo `results.csv` con los valores de acurracy obtenidos para cada combinacion de hyperparametros corrida.
Si se corre con `-s` se guarda ademas un archivo con el mejor modelo obtenido. El formato de este archivo consiste en:

- Una primera linea con la cantidad de palabras N 
- Una segunda linea con un entero M donde MxM es el tamaño de la grilla del modelo.
- Las lineas siguientes corresponden al tensor de pesos, guardado como M veces M lineas de N (M, M, N)
- Finalmente se guarda la matriz de categorias de MxM para poder indicar la categoria de nuevos casos 

Si se corre con `g` se guardan graficos de la evolucion del entrenamiento y de las categorias obtenidas para conjuntos de entrenamiento y validacion en un mapeo sobre la grilla.


