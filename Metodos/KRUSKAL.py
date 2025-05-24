import pandas as pd
from scipy.stats import kruskal

def calcular_kruskal_paso(*grupos):
    valores = []
    etiquetas = []

    for i, grupo in enumerate(grupos):
        valores += grupo
        etiquetas += [f"G{i+1}"] * len(grupo)

    df = pd.DataFrame({
        "valor": valores,
        "grupo": etiquetas
    })

    # Asignamos rangos globales
    df["rango"] = df["valor"].rank(method="average")

    n_total = len(df)
    H = 0
    for grupo in df["grupo"].unique():
        n_i = len(df[df["grupo"] == grupo])
        R_i = df[df["grupo"] == grupo]["rango"].sum()
        H += (R_i**2) / n_i

    H = (12 / (n_total * (n_total + 1))) * H - 3 * (n_total + 1)

    return H

def calcular_kruskal_directo(*grupos):
    stat, p = kruskal(*grupos)
    return stat, p

if __name__ == "__main__":
    grupo_1 = [12, 15, 14, 10, 13]
    grupo_2 = [22, 25, 24, 23, 20]
    grupo_3 = [30, 32, 29, 31, 28]

    print("Kruskal paso a paso:", calcular_kruskal_paso(grupo_1, grupo_2, grupo_3))
    stat, p = calcular_kruskal_directo(grupo_1, grupo_2, grupo_3)
    print("Kruskal scipy directo:", stat, p)
