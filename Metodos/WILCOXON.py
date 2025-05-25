import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def calcular_wilcoxon_paso(grupo_1, grupo_2) -> float:
    
    # Diferencias pareadas
    diferencias = np.array(grupo_1) - np.array(grupo_2)
    
    # Excluir ceros (diferencias nulas)
    diferencias = diferencias[diferencias != 0]
    
    # Valor absoluto y signo de las diferencias
    abs_diff = np.abs(diferencias)
    signos = np.sign(diferencias)
    
    # Asignar rangos a los valores absolutos
    rangos = pd.Series(abs_diff).rank(method="average")
    
    # Calcular la suma de rangos positivos y negativos
    W_pos = sum(rangos[signos > 0])
    W_neg = sum(rangos[signos < 0])
    
    # Estadístico de Wilcoxon es el mínimo de ambas sumas
    W = min(W_pos, W_neg)
    
    return W


def calcular_wilcoxon(grupo_1, grupo_2) -> float:
    stat, p = wilcoxon(grupo_1, grupo_2)
    return stat,p

if __name__ == "__main__":
    grupo_1 = [20, 23, 21, 25, 18, 19, 17]
    grupo_2 = [30, 35, 28, 32, 33, 29, 31]
    
    print(calcular_wilcoxon(grupo_1,grupo_2))
    print(calcular_wilcoxon_paso(grupo_1,grupo_2))