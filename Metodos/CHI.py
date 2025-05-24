import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

def calcular_chi_paso(*args) -> tuple:
    """
    Calcula el estadístico Chi-cuadrado de Pearson paso a paso (sin corrección de Yates).
    
    Parámetros:
        args: listas numéricas que representan las filas de una tabla de contingencia.

    Retorna:
        chi_total (float): estadístico chi-cuadrado.
        gl (int): grados de libertad.
    """
    # Crear DataFrame desde las filas
    df = pd.DataFrame(args)

    # Totales marginales
    total_fila = df.sum(axis=1)
    total_columna = df.sum(axis=0)
    total_general = df.values.sum()

    # Frecuencias esperadas
    esperadas = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for fila in df.index:
        for col in df.columns:
            esperadas.loc[fila, col] = (total_fila[fila] * total_columna[col]) / total_general

    # Cálculo del chi-cuadrado
    chi_celda = (df - esperadas) ** 2 / esperadas
    chi_total = chi_celda.values.sum()

    # Grados de libertad
    gl = (df.shape[0] - 1) * (df.shape[1] - 1)

    return chi_total, gl

def calcular_chi(*args) -> tuple:
    """
    Calcula el estadístico Chi-cuadrado de Pearson con Scipy (sin corrección).

    Parámetros:
        args: listas numéricas que representan las filas de una tabla de contingencia.

    Retorna:
        chi2 (float): estadístico chi-cuadrado.
        p (float): valor p.
        dof (int): grados de libertad.
        esperada (ndarray): matriz de frecuencias esperadas.
    """
    chi2, p, dof, esperada = chi2_contingency(args, correction=False)
    return chi2, p, dof, esperada

if __name__ == "__main__":
    # Datos de ejemplo
    fila_1 = [20, 15]
    fila_2 = [30, 35]

    print("📊 Método Paso a Paso:")
    chi_manual, gl_manual = calcular_chi_paso(fila_1, fila_2)
    print(f"Chi-cuadrado: {chi_manual:.4f}")
    print(f"Grados de libertad: {gl_manual}\n")

    print("⚙️ Método con Scipy:")
    chi2, p, dof, esperada = calcular_chi(fila_1, fila_2)
    print(f"Chi-cuadrado: {chi2:.4f}")
    print(f"p-valor: {p:.4f}")
    print(f"Grados de libertad: {dof}")
    print(f"Frecuencia esperada:\n{pd.DataFrame(esperada)}")
