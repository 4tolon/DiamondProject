import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
import seaborn as sns
from matplotlib import pyplot as plt

def magia(train, test):
    """
    Función para procesar los dos data sets, train y test de diamantes
    arg:
    train, test
    ret: 
    X, y, X_train, X_test, y_train, y_test
    """ 

    cambia_color = {'G':4, 'H':5, 'F':3, 'J':7, 'E':2, 'I':6, 'D':1}
    cambia_clarity = {'VVS2':3, 'VS2':5, 'VS1':4, 'SI2':7, 'SI1':6, 'VVS1':2, 'IF':1, 'I1':8}
    cambia_cut = {"Ideal":1, "Premium":2,"Very Good":3,"Good":4, "Fair":5}
    
    train.color = train.color.replace(cambia_color)
    train.clarity = train.clarity.replace(cambia_clarity)
    train.cut = train.cut.replace(cambia_cut)

    test.color = test.color.replace(cambia_color)
    test.clarity = test.clarity.replace(cambia_clarity)
    test.cut = test.cut.replace(cambia_cut)

    X = train.drop("price", axis=1)
    X =  X.drop('id', axis = 1)
    y = train.price 

    X_train, X_test, y_train, y_test = tts(X,y, test_size=0.2)

    return X, y, X_train, X_test, y_train, y_test


def get_iqr_values(df_in, col_name):
    """
    Función obterner datos estadisticos de columnas de un dataframe.
    arg:
    dataframe, nombre_de_columna
    ret: 
    median, q1, q3, iqr, minimum, maximum
    """
    median = df_in[col_name].median()
    q1 = df_in[col_name].quantile(0.25) # 1er quartil
    q3 = df_in[col_name].quantile(0.75) # 3er quartil
    iqr = q3-q1 #rango intercuartile
    minimum  = q1-1.5*iqr 
    maximum = q3+1.5*iqr 
    return median, q1, q3, iqr, minimum, maximum

def get_iqr_text(df_in, col_name):
    """
    Funcion interna para convertir en string todos los datos obtenidos
    """
    median, q1, q3, iqr, minimum, maximum = get_iqr_values(df_in, col_name)
    text = f"median={median:.2f}, q1={q1:.2f}, q3={q3:.2f}, iqr={iqr:.2f}, minimum={minimum:.2f}, maximum={maximum:.2f}"
    return text

def borra_outliers(df_in, col_name):
    
    """
    Funcion para borra los datos que no se encuentren entre el max y el min del boxplort
    arg:
    dataframe, nombre_de_columna
    ret: 
    dataframe sin outliers
    """
    _, _, _, _, minimum, maximum = get_iqr_values(df_in, col_name)
    df_out = df_in.loc[(df_in[col_name] > minimum) & (df_in[col_name] < maximum)]
    return df_out

def cuenta_outliers(df_in, col_name):
    
    """
    Cuenta el nummero de outliers 
    arg:
    dataframe, nombre_de_columna
    ret: 
    numero de ountliers 
    """
    _, _, _, _, minimum, maximum = get_iqr_values(df_in, col_name)
    df_outliers = df_in.loc[(df_in[col_name] <= minimum) | (df_in[col_name] >= maximum)]
    return df_outliers.shape[0]

def box_plot(df_in, col_name):

    """
    Pinta boxplot
    arg:
    dataframe, nombre_de_columna
    ret: 
    grafica 
    """
    title = get_iqr_text(df_in, col_name)
    sns.boxplot(x= df_in[col_name], color = 'g')
    plt.title(title)
    plt.show()