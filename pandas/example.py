import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

def clean_dataset(dataframe):
    assert isinstance(dataframe, pd.DataFrame)
    dataframe.dropna(inplace=True)
    indices_to_keep = dataframe.isin([np.nan, np.inf, -np.inf]).any(1)
    return dataframe[indices_to_keep].astype(np.float64)

dataframe = pd.read_csv('example.csv')
print(dataframe)

#en esta funcion obtenemos un csv con datos de ejemplos y limpiamos la data, quitamos los NAN (porque nos puede dar error)
#
# recorro las columnas para chequear cuantos valores nulos existen por cada una osea los IS NAN (NaN)
for column in list(dataframe.columns):
    print('{}: {}'.format(column, dataframe[column].isnull().sum()))


# Creo un objeto de pandas solo con ciertos datos en este caso, columnas numericas 
dataframe_res = dataframe.loc[:, dataframe.select_dtypes(include=[np.number]).columns]
print(dataframe_res)


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i +n]

_colums = list(filter(lambda x: x != 'Id', dataframe.columns))
#obtengo la descripcion de cada columna

print("las columnas",  _colums)

#uso el divide chunks para retornar los datos de cada uno de los sub array de elemtnos
""" for columns in divide_chunks(_colums, 5):
    plt.figure(figsize=(10,6))
    dataframe.boxplot(columns) """

## para mostrar graficos de datos usando  matpotlib aun en proceso de aprendizaje
#  no tengo tanto conocimiento en esta area de graficos
# """

#para encontrar correlacion de datos
print("correlaciones entre datos", dataframe.corr())


