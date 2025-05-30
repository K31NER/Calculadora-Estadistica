import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

def calcular_mannwhitneyu_paso(grupo_1:list ,grupo_2:list) -> float:
    """ Calcula la U paso a paso

    Parametros: \n
    grupo 1 : lista \n
    grupo 2 : lista
    """
    # Calculamos el tamaño de los grupos
    tamaño_1 , tamaño_2 = len(grupo_1), len(grupo_2)
    
    # definimos el dataframe
    df = pd.DataFrame({
        "valor": grupo_1 + grupo_2,
        "grupo": ["A"] * tamaño_1 + ["B"]* tamaño_2
    })
    
    # Calculamos los rangos
    df["rango"] = df["valor"].rank(method="average") # Asiganamos rango en general
    
    # filtramos por grupo
    rango_1 = df[df["grupo"] == "A"]["rango"].sum()
    rango_2 = df[df["grupo"] == "B"]["rango"].sum()
    
    # Aplicamos la formula
    U1 = (tamaño_1*tamaño_2) + ((tamaño_1*(tamaño_1 +1))/2) - rango_1
    U2 = (tamaño_1*tamaño_2) + ((tamaño_2*(tamaño_2 +1))/2) - rango_2
    
    # Retornamos el menor de lo valores
    U_menor = min(U1,U2)
    
    return U_menor

def calcular_mannwhitneyu(grupo_1:list ,grupo_2:list) -> float:
    
    """ Calcula la U de manera directa

    Parametros: \n
    grupo 1 : lista \n
    grupo 2 : lista
    """
    U, P = mannwhitneyu(grupo_1,grupo_2, alternative='two-sided')
    return U, P

if __name__ == "__main__":
    grupo_1 = [20, 23, 21, 25, 18, 19, 17]
    grupo_2 = [30, 35, 28, 32, 33, 29, 31]
    
    print("Paso a paso")
    print(calcular_mannwhitneyu_paso(grupo_1,grupo_2))
    print("")
    print("Directo")
    u = calcular_mannwhitneyu(grupo_1,grupo_2)
    print(u)