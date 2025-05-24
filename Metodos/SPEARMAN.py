import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def calcular_coeficiente_paso(x: list, y :list) -> tuple:
    """Funcion para calcular el coeficiente de Spearman entre dos listas."""
    # Obtenemos los 2 rangos de las listas
    X = np.array(x)
    Y = np.array(y) 
    if len(x) != len(y):
        raise ValueError("Las listas deben tener la misma longitud.")
    
    try:
        # Volvemos los datos en un dataframe
        df = pd.DataFrame({'X': X, 'Y': Y})
        
        # Calculamos los rangos de cada lista 
        df['rango_x'] = df['X'].rank()
        df['rango_y'] = df['Y'].rank()
        
        # Calculamos la diferencia entre los rangos
        df['dᵢ'] = df['rango_x'] - df['rango_y']
        
        # Calculamos el cuadrado de la diferencias
        df['d²'] = df['dᵢ'] ** 2
        
        suma_diferencias = df['d²'].sum()
        
        # Calculamos el coeficiente de spearman con su formula
        spearman = 1 - (6 * suma_diferencias) / (len(df["X"]) * (len(df["X"])**2 - 1))
        
        # Devolvemos la tabla y el coeficiente
        return df , spearman
    except Exception as e:
        raise ValueError(f"Error al calcular los rangos: {e}")

def calular_spearman(x: list, y: list) -> tuple:
    """ Calcula el coeficiente de correlacion de spearman de forma directa """
    
    X = np.array(x)
    Y = np.array(y)
    
    rho, p_valor = spearmanr(X,Y)
    
    return rho, p_valor

if __name__ == "__main__":
    
    x = [4, 6, 2, 8, 5]
    y = [78, 85, 72, 88, 80]
    
    rho, p = calular_spearman(x,y)
    
    print(f"spearman : {rho}")
    print(f"P valor: {p}")