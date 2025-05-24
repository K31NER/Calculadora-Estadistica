from scipy.stats import kendalltau

def calcular_kendall_tau_paso(x, y) -> float:
    """ Calcula el tau de kendall """
    assert len(x) == len(y), "Las listas deben tener la misma longitud"
    n = len(x)
    C = D = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            # Producto de diferencias
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            prod = dx * dy
            if prod > 0:
                C += 1  # Concordante
            elif prod < 0:
                D += 1  # Discordante
            # Si prod == 0, es empate en alguna dimensión y no se cuenta

    tau = (C - D) / (n * (n - 1) / 2)
    return tau

def calcular_kendall_tau(x,y) -> tuple:
    """ Calcula el tau de kendall de forma directa """
    assert len(x) == len(y) , "Las listas deben tener la misma longitud"
    tau,p_valor = kendalltau(x,y)
    return tau , p_valor

if __name__ == "__main__":
    x = [12, 2, 1, 12, 2]
    y = [1, 4, 7, 1, 0]
    
    print("Metodo manual")
    print(calcular_kendall_tau_paso(x,y))
    
    print("Metodo directo")
    print(calcular_kendall_tau(x,y))