import numpy as np
from scipy.stats import f_oneway

# Datos de ejemplo para tres grupos
grupo_1 = [15, 16, 17, 18, 19]
grupo_2 = [25, 26, 27, 28, 29]
grupo_3 = [35, 36, 37, 38, 39]

lista = [grupo_1,grupo_2,grupo_3]

def calular_anova_pasos(lista_grupos:list) -> tuple:
    """ Funcion para caluclar el ANOVA de X nuemero de grupos \n
        Recibe una lista de listas
    """
    
    # Paso 1 : Calcular los promedios
    # Sacamos la medias de cada grupo
    medias_grupo  = list([np.mean(grupo) for grupo in lista_grupos])
    # Media global
    media_global = np.mean([x for grupo in lista_grupos for x in grupo])
    
    # Paso 2: SCB (Suma de cuadrados entre grupos)
    SCB = sum(len(g) * (np.mean(g) - media_global)**2 for g in lista_grupos)
    
    # Paso 3: SCD (Suma de cuadrados dentro de los grupos)
    SCD = sum(sum((x - np.mean(g))**2 for x in g) for g in lista_grupos)
    
    # Paso 4: Suma total de cuadrados
    SCT = SCB + SCD
    
    # Paso 5: Grados de libertad
    k = len(lista_grupos)
    n = sum(len(g) for g in lista_grupos)
    
    glB = k-1
    glD = n-k
    
    # Paso 6: Cuadrados medios
    MSB = SCB / glB
    MSD = SCD / glD
    
    # Paso 7: Estadistico F 
    F = MSB / MSD
    
    return SCB, SCD, F
    

def calcular_anova(lista_grupos:list) -> tuple:
    
    """ Calulo de anova de manera directa """
    
    # Calculamos F y P
    F,P = f_oneway(*lista_grupos) # Desempaquetamos la lista con *
    
    return F,P


if __name__ == "__main__":
    
    metodo_1 = calular_anova_pasos(lista)

    SCB , SCD , F = metodo_1

    print(f"SCB: {SCB}")
    print(f"SCD: {SCD}")
    print(f"F: {F}")

    print("Metodo 2")
    print("".ljust(100,"-"))
    metodo_2 = calcular_anova(lista)
    F,P = metodo_2
    print(f"F: {F}")
    print(f"P: {P}")