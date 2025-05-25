import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

try:
    from Metodos.ANOVA import calular_anova_pasos, calcular_anova
    from Metodos.CHI import calcular_chi_paso, calcular_chi
    from Metodos.KENDALL import calcular_kendall_tau_paso, calcular_kendall_tau
    from Metodos.KRUSKAL import calcular_kruskal_paso, calcular_kruskal_directo
    from Metodos.MANNWHITNEY import calcular_mannwhitneyu_paso, calcular_mannwhitneyu # Asegúrate del nombre del archivo
    from Metodos.SPEARMAN import calcular_coeficiente_paso, calular_spearman
    from Metodos.WILCOXON import calcular_wilcoxon_paso, calcular_wilcoxon
except ImportError as e:
    st.error(f"Error al importar módulos desde Metodos: {e}. "
             "Verifica nombres de archivo (ej: 'MANNWHITNEYpy.py') y que estén en la carpeta 'Metodos'.")
    st.stop()

def parse_input_list(input_str: str):
    if not input_str.strip(): return []
    try:
        return [float(x.strip()) for x in input_str.split(',') if x.strip()]
    except ValueError:
        return None

# --- Funciones de Graficación (sin cambios) ---
def graficar_boxplot_grupos(lista_de_grupos: list, nombres_grupos: list = None, titulo_graf="Distribución de Valores por Grupo"):
    if not lista_de_grupos or not all(isinstance(g, list) and g for g in lista_de_grupos):
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    data_long = []
    for i, grupo_data in enumerate(lista_de_grupos):
        nombre = nombres_grupos[i] if nombres_grupos and i < len(nombres_grupos) else f"Grupo {i+1}"
        for valor in grupo_data: data_long.append({'Grupo': nombre, 'Valor': valor})
    df_long = pd.DataFrame(data_long)
    if not df_long.empty:
        sns.boxplot(x='Grupo', y='Valor', data=df_long, ax=ax, palette="Set2", showfliers=True)
        sns.stripplot(x='Grupo', y='Valor', data=df_long, ax=ax, color=".25", size=4, jitter=True)
        ax.set_title(titulo_graf)
        ax.set_xlabel('Grupo')
        ax.set_ylabel('Valores')
        plt.tight_layout()
        return fig
    return None

def graficar_scatter(x_data: list, y_data: list, x_label: str = "Variable X", y_label: str = "Variable Y", titulo: str = "Diagrama de Dispersión"):
    if not x_data or not y_data or len(x_data) != len(y_data): return None
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=x_data, y=y_data, ax=ax, s=80, alpha=0.8, hue=None)
    ax.set_title(titulo)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.tight_layout()
    return fig

def graficar_tabla_contingencia(df_observada: pd.DataFrame, df_esperada: pd.DataFrame = None, titulo_obs="Frecuencias Observadas"):
    if df_observada.empty: return None
    num_plots = 1
    if df_esperada is not None and not df_esperada.empty: num_plots = 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), squeeze=False)
    ax_obs = axes[0,0]
    sns.heatmap(df_observada, annot=True, fmt=".1f", cmap="Blues", ax=ax_obs, cbar=True)
    ax_obs.set_title(titulo_obs)
    ax_obs.set_xlabel("Columnas")
    ax_obs.set_ylabel("Filas")
    if num_plots > 1 and df_esperada is not None and not df_esperada.empty:
        ax_esp = axes[0,1]
        sns.heatmap(df_esperada, annot=True, fmt=".2f", cmap="Oranges", ax=ax_esp, cbar=True)
        ax_esp.set_title("Frecuencias Esperadas")
        ax_esp.set_xlabel("Columnas")
        ax_esp.set_ylabel("Filas")
    plt.tight_layout()
    return fig

def graficar_diferencias_wilcoxon(grupo1: list, grupo2: list):
    if not grupo1 or not grupo2 or len(grupo1) != len(grupo2): return None
    diferencias = np.array(grupo1) - np.array(grupo2)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(diferencias, kde=True, ax=ax, color="purple", bins='auto')
    ax.axvline(np.mean(diferencias), color='red', linestyle='--', label=f"Media Dif: {np.mean(diferencias):.2f}")
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title('Histograma de las Diferencias (Grupo1 - Grupo2)')
    ax.set_xlabel('Diferencia')
    ax.set_ylabel('Frecuencia')
    ax.legend()
    plt.tight_layout()
    return fig

st.set_page_config(layout="wide")
st.title("📊 Calculadora Estadística con Gráficos")

st.sidebar.header("Seleccionar Método Estadístico")
metodo_seleccionado = st.sidebar.selectbox(
    "Método:",
    ["ANOVA", "Chi-cuadrado", "Tau de Kendall", "Kruskal-Wallis", "U de Mann-Whitney", "Rho de Spearman", "Wilcoxon (muestras pareadas)"],
    key="metodo_selector", help="Elige la prueba estadística."
)

# --- Sección de ANOVA ---
if metodo_seleccionado == "ANOVA":
    st.header("Análisis de Varianza (ANOVA)")
    st.markdown("Compara las medias de tres o más grupos. Ejemplo: Grupo 1: `15,16,17,18,19`, Grupo 2: `25,26,27,28,29`")
    num_grupos_anova = st.number_input("Número de grupos:", min_value=2, value=2, step=1, key="anova_num_grupos")
    datos_grupos_anova_str = []
    num_cols_display = int(num_grupos_anova) if int(num_grupos_anova) <= 4 else 1
    cols_anova = st.columns(num_cols_display)
    for i in range(int(num_grupos_anova)):
        with cols_anova[i % num_cols_display]:
            datos_grupos_anova_str.append(st.text_input(f"Datos Grupo {i+1}", key=f"anova_grupo_{i}", help="Números separados por comas"))

    if st.button("Calcular ANOVA", key="anova_calc"):
        lista_grupos_anova_calc = [parse_input_list(s) for s in datos_grupos_anova_str]
        valid_input = not any(g is None for g in lista_grupos_anova_calc) and not any(not g for g in lista_grupos_anova_calc)
        
        if not valid_input: st.error("Entrada inválida o grupo vacío. Revisa los datos.")
        elif len(lista_grupos_anova_calc) < 2: st.error("Se necesitan al menos dos grupos.")
        else:
            try:
                st.subheader("Visualización de los Datos (ANOVA)")
                fig_anova = graficar_boxplot_grupos(lista_grupos_anova_calc, titulo_graf="Distribución para ANOVA")
                if fig_anova: st.pyplot(fig_anova)
                else: st.write("No se pudo generar el gráfico (datos insuficientes o inválidos).")

                st.subheader("Resultados del Cálculo Paso a Paso (ANOVA):")
                # Asume que calular_anova_pasos retorna una tupla (SCB, SCD, F)
                scb, scd, f_pasos = calular_anova_pasos(lista_grupos_anova_calc)
                st.metric(label="SCB (Suma de Cuadrados Entre Grupos)", value=f"{scb:.4f}")
                st.metric(label="SCD (Suma de Cuadrados Dentro de Grupos)", value=f"{scd:.4f}")
                st.metric(label="Estadístico F (pasos)", value=f"{f_pasos:.4f}")

                st.subheader("Resultados del Cálculo Directo con SciPy (ANOVA):")
                f_directo, p_directo = calcular_anova(lista_grupos_anova_calc)
                st.metric(label="Estadístico F (directo)", value=f"{f_directo:.4f}")
                st.metric(label="Valor p", value=f"{p_directo:.4f}")
                if p_directo < 0.05: st.success(f"Significativo (p = {p_directo:.4f})")
                else: st.info(f"No significativo (p = {p_directo:.4f})")
            except Exception as e: st.error(f"Error en ANOVA: {e}")

# --- Sección de Chi-cuadrado ---
elif metodo_seleccionado == "Chi-cuadrado":
    st.header("Prueba de Chi-cuadrado de Independencia")
    st.markdown("Evalúa asociación. Ejemplo: Fila 1: `20,15`, Fila 2: `30,35`")
    num_filas_chi = st.number_input("Número de filas:", min_value=2, value=2, step=1, key="chi_num_filas")
    datos_filas_chi_str = [st.text_input(f"Fila {i+1}", key=f"chi_fila_{i}") for i in range(int(num_filas_chi))]
    
    if st.button("Calcular Chi-cuadrado", key="chi_calc"):
        filas_chi_calc = [parse_input_list(s) for s in datos_filas_chi_str]
        valid_input = not any(f is None for f in filas_chi_calc) and not any(not f for f in filas_chi_calc)
        num_cols_expected = len(filas_chi_calc[0]) if valid_input and filas_chi_calc else -1
        consistent_cols = all(len(f) == num_cols_expected for f in filas_chi_calc) if valid_input and num_cols_expected > 0 else False

        if not valid_input: st.error("Entrada inválida o fila vacía.")
        elif not consistent_cols or num_cols_expected < 2 : st.error("Todas las filas deben tener el mismo número de columnas (mínimo 2).")
        elif len(filas_chi_calc) < 2: st.error("Se necesitan al menos dos filas.")
        else:
            try:
                df_observada_chi = pd.DataFrame(filas_chi_calc)
                df_esperada_chi = None
                try: 
                    _, _, _, esperada_raw = calcular_chi(*[list(f) for f in filas_chi_calc])
                    df_esperada_chi = pd.DataFrame(esperada_raw)
                except: pass

                st.subheader("Visualización de Frecuencias (Chi-cuadrado)")
                fig_chi = graficar_tabla_contingencia(df_observada_chi, df_esperada_chi)
                if fig_chi: st.pyplot(fig_chi)
                else: st.write("No se pudo generar el gráfico.")
                
                st.subheader("Resultados del Cálculo Paso a Paso (Chi-cuadrado):")
                # Asume que calcular_chi_paso retorna (chi_total, gl)
                chi_p, gl_p = calcular_chi_paso(*[list(f) for f in filas_chi_calc])
                st.metric(label="Chi-cuadrado (pasos)", value=f"{chi_p:.4f}")
                st.metric(label="Grados de libertad (pasos)", value=f"{gl_p}")

                st.subheader("Resultados del Cálculo Directo con SciPy (Chi-cuadrado):")
                chi_directo, p_directo, gl_directo, _ = calcular_chi(*[list(f) for f in filas_chi_calc])
                st.metric(label="Chi-cuadrado (directo)", value=f"{chi_directo:.4f}")
                st.metric(label="Valor p", value=f"{p_directo:.4f}")
                st.metric(label="Grados de libertad (directo)", value=f"{gl_directo}")
                if p_directo < 0.05: st.success(f"Significativo (p = {p_directo:.4f})")
                else: st.info(f"No significativo (p = {p_directo:.4f})")
            except Exception as e: st.error(f"Error en Chi-cuadrado: {e}")

# --- Sección de Tau de Kendall ---
elif metodo_seleccionado == "Tau de Kendall":
    st.header("Coeficiente de Correlación Tau de Kendall")
    st.markdown("Mide correlación ordinal. Ejemplo X: `12,2,1,12,2`, Y: `1,4,7,1,0`")
    col1_k, col2_k = st.columns(2)
    with col1_k: datos_x_k_str = st.text_input("Grupo X", key="k_x")
    with col2_k: datos_y_k_str = st.text_input("Grupo Y", key="k_y")

    if st.button("Calcular Tau de Kendall", key="k_calc"):
        x_k = parse_input_list(datos_x_k_str)
        y_k = parse_input_list(datos_y_k_str)
        valid_input = x_k is not None and y_k is not None and x_k and y_k and len(x_k) == len(y_k)

        if not valid_input: st.error("Entrada inválida, grupos vacíos o longitudes diferentes.")
        else:
            try:
                st.subheader("Visualización de los Datos (Kendall)")
                fig_kendall = graficar_scatter(x_k, y_k, "Valores X", "Valores Y", "Dispersión para Tau de Kendall")
                if fig_kendall: st.pyplot(fig_kendall)
                else: st.write("No se pudo generar el gráfico.")

                st.subheader("Resultados del Cálculo Paso a Paso (Tau de Kendall):")
                # Asume que calcular_kendall_tau_paso retorna el valor de tau directamente
                tau_pasos_kendall = calcular_kendall_tau_paso(x_k, y_k)
                st.metric(label="Tau de Kendall (pasos)", value=f"{tau_pasos_kendall:.4f}")
                
                st.subheader("Resultados del Cálculo Directo con SciPy (Tau de Kendall):")
                tau_directo, p_directo = calcular_kendall_tau(x_k, y_k)
                st.metric(label="Tau de Kendall (directo)", value=f"{tau_directo:.4f}")
                st.metric(label="Valor p", value=f"{p_directo:.4f}")
                if p_directo < 0.05: st.success(f"Significativo (p = {p_directo:.4f})")
                else: st.info(f"No significativo (p = {p_directo:.4f})")
            except Exception as e: st.error(f"Error en Tau de Kendall: {e}")

# --- Sección de Kruskal-Wallis ---
elif metodo_seleccionado == "Kruskal-Wallis":
    st.header("Prueba de Kruskal-Wallis")
    st.markdown("Alternativa no paramétrica a ANOVA. Ejemplo G1: `12,15,14`, G2: `22,25,24`")
    num_grupos_kw = st.number_input("Número de grupos:", min_value=2, value=2, step=1, key="kw_num_grupos")
    datos_grupos_kw_str = []
    num_cols_display_kw = int(num_grupos_kw) if int(num_grupos_kw) <= 4 else 1
    cols_kw = st.columns(num_cols_display_kw)
    for i in range(int(num_grupos_kw)):
        with cols_kw[i % num_cols_display_kw]:
            datos_grupos_kw_str.append(st.text_input(f"Datos Grupo {i+1}", key=f"kw_grupo_{i}"))

    if st.button("Calcular Kruskal-Wallis", key="kw_calc"):
        lista_grupos_kw_calc = [parse_input_list(s) for s in datos_grupos_kw_str]
        valid_input = not any(g is None for g in lista_grupos_kw_calc) and not any(not g for g in lista_grupos_kw_calc)

        if not valid_input: st.error("Entrada inválida o grupo vacío.")
        elif len(lista_grupos_kw_calc) < 2: st.error("Se necesitan al menos dos grupos.")
        else:
            try:
                st.subheader("Visualización de los Datos (Kruskal-Wallis)")
                fig_kw = graficar_boxplot_grupos(lista_grupos_kw_calc, titulo_graf="Distribución para Kruskal-Wallis")
                if fig_kw: st.pyplot(fig_kw)
                else: st.write("No se pudo generar el gráfico.")

                st.subheader("Resultados del Cálculo Paso a Paso (Kruskal-Wallis):")
                # Asume que calcular_kruskal_paso retorna el valor H directamente
                h_pasos_kw = calcular_kruskal_paso(*lista_grupos_kw_calc)
                st.metric(label="Estadístico H (pasos)", value=f"{h_pasos_kw:.4f}")

                st.subheader("Resultados del Cálculo Directo con SciPy (Kruskal-Wallis):")
                h_directo, p_directo = calcular_kruskal_directo(*lista_grupos_kw_calc)
                st.metric(label="Estadístico H (directo)", value=f"{h_directo:.4f}")
                st.metric(label="Valor p", value=f"{p_directo:.4f}")
                if p_directo < 0.05: st.success(f"Significativo (p = {p_directo:.4f})")
                else: st.info(f"No significativo (p = {p_directo:.4f})")
            except Exception as e: st.error(f"Error en Kruskal-Wallis: {e}")

# --- Sección de U de Mann-Whitney ---
elif metodo_seleccionado == "U de Mann-Whitney":
    st.header("Prueba U de Mann-Whitney")
    st.markdown("Compara medianas de dos grupos independientes. Ejemplo G1: `20,23,21`, G2: `30,35,28`")
    col1_mw, col2_mw = st.columns(2)
    with col1_mw: datos_g1_mw_str = st.text_input("Grupo 1", key="mw_g1")
    with col2_mw: datos_g2_mw_str = st.text_input("Grupo 2", key="mw_g2")

    if st.button("Calcular U de Mann-Whitney", key="mw_calc"):
        g1_mw = parse_input_list(datos_g1_mw_str)
        g2_mw = parse_input_list(datos_g2_mw_str)
        valid_input = g1_mw is not None and g2_mw is not None and g1_mw and g2_mw

        if not valid_input: st.error("Entrada inválida o grupo vacío.")
        else:
            try:
                st.subheader("Visualización de los Datos (Mann-Whitney)")
                fig_mw = graficar_boxplot_grupos([g1_mw, g2_mw], nombres_grupos=["Grupo 1", "Grupo 2"], titulo_graf="Distribución para Mann-Whitney U")
                if fig_mw: st.pyplot(fig_mw)
                else: st.write("No se pudo generar el gráfico.")

                st.subheader("Resultados del Cálculo Paso a Paso (U de Mann-Whitney):")
                # Asume que calcular_mannwhitneyu_paso retorna el valor U directamente
                u_pasos_mw = calcular_mannwhitneyu_paso(g1_mw, g2_mw)
                st.metric(label="Estadístico U (pasos)", value=f"{u_pasos_mw}")
                
                st.subheader("Resultados del Cálculo Directo con SciPy (U de Mann-Whitney):")
                # Asegúrate que tu función en Metodos/MANNWHITNEYpy.py retorne (U, P)
                u_directo, p_directo = calcular_mannwhitneyu(g1_mw, g2_mw)
                st.metric(label="Estadístico U (directo)", value=f"{u_directo}")
                st.metric(label="Valor p", value=f"{p_directo:.4f}")
                if p_directo < 0.05: st.success(f"Significativo (p = {p_directo:.4f})")
                else: st.info(f"No significativo (p = {p_directo:.4f})")
            except Exception as e: st.error(f"Error en U de Mann-Whitney: {e}")

# --- Sección de Rho de Spearman ---
elif metodo_seleccionado == "Rho de Spearman":
    st.header("Coeficiente de Correlación Rho de Spearman")
    st.markdown("Mide correlación monotónica. Ejemplo X: `4,6,2`, Y: `78,85,72`")
    col1_s, col2_s = st.columns(2)
    with col1_s: datos_x_s_str = st.text_input("Grupo X", key="s_x")
    with col2_s: datos_y_s_str = st.text_input("Grupo Y", key="s_y")

    if st.button("Calcular Rho de Spearman", key="s_calc"):
        x_s = parse_input_list(datos_x_s_str)
        y_s = parse_input_list(datos_y_s_str)
        valid_input = x_s is not None and y_s is not None and x_s and y_s and len(x_s) == len(y_s)

        if not valid_input: st.error("Entrada inválida, grupos vacíos o longitudes diferentes.")
        else:
            try:
                st.subheader("Visualización de los Datos (Spearman)")
                fig_spearman = graficar_scatter(x_s, y_s, "Valores X", "Valores Y", "Dispersión para Rho de Spearman")
                if fig_spearman: st.pyplot(fig_spearman)
                else: st.write("No se pudo generar el gráfico.")

                st.subheader("Resultados del Cálculo Paso a Paso (Rho de Spearman):")
                # Asume que calcular_coeficiente_paso retorna (DataFrame, rho)
                df_pasos_spearman, rho_pasos_spearman = calcular_coeficiente_paso(x_s, y_s)
                st.write("Tabla de Cálculos Intermedios (pasos):")
                st.dataframe(df_pasos_spearman)
                st.metric(label="Rho de Spearman (pasos)", value=f"{rho_pasos_spearman:.4f}")
                
                st.subheader("Resultados del Cálculo Directo con SciPy (Rho de Spearman):")
                rho_directo, p_directo = calular_spearman(x_s, y_s)
                st.metric(label="Rho de Spearman (directo)", value=f"{rho_directo:.4f}")
                st.metric(label="Valor p", value=f"{p_directo:.4f}")
                if p_directo < 0.05: st.success(f"Significativo (p = {p_directo:.4f})")
                else: st.info(f"No significativo (p = {p_directo:.4f})")
            except Exception as e: st.error(f"Error en Rho de Spearman: {e}")

# --- Sección de Wilcoxon (muestras pareadas) ---
elif metodo_seleccionado == "Wilcoxon (muestras pareadas)":
    st.header("Prueba de Wilcoxon para Muestras Pareadas")
    st.markdown("Compara dos muestras pareadas. Ejemplo G1: `20,23,21`, G2: `30,35,28`")
    col1_w, col2_w = st.columns(2)
    with col1_w: datos_g1_w_str = st.text_input("Grupo 1 (Antes)", key="w_g1")
    with col2_w: datos_g2_w_str = st.text_input("Grupo 2 (Después)", key="w_g2")

    if st.button("Calcular Wilcoxon", key="w_calc"):
        g1_w = parse_input_list(datos_g1_w_str)
        g2_w = parse_input_list(datos_g2_w_str)
        valid_input = g1_w is not None and g2_w is not None and g1_w and g2_w and len(g1_w) == len(g2_w)

        if not valid_input: st.error("Entrada inválida, grupos vacíos o longitudes diferentes.")
        else:
            try:
                st.subheader("Visualización de las Diferencias (Wilcoxon)")
                fig_wilcoxon = graficar_diferencias_wilcoxon(g1_w, g2_w)
                if fig_wilcoxon: st.pyplot(fig_wilcoxon)
                else: st.write("No se pudo generar el gráfico.")

                st.subheader("Resultados del Cálculo Paso a Paso (Wilcoxon):")
                # Asume que calcular_wilcoxon_paso retorna el valor W directamente
                w_pasos_wilcoxon = calcular_wilcoxon_paso(g1_w, g2_w)
                st.metric(label="Estadístico W (pasos)", value=f"{w_pasos_wilcoxon}")

                st.subheader("Resultados del Cálculo Directo con SciPy (Wilcoxon):")
                # Asegúrate que tu función en Metodos/WILCOXON.py retorne (W, P)
                w_directo, p_directo = calcular_wilcoxon(g1_w, g2_w) 
                if np.isnan(w_directo): st.warning("Cálculo directo no posible (ej: todas las diferencias son cero).")
                else:
                    st.metric(label="Estadístico W (directo)", value=f"{w_directo}")
                    st.metric(label="Valor p", value=f"{p_directo:.4f}")
                    if p_directo < 0.05: st.success(f"Significativo (p = {p_directo:.4f})")
                    else: st.info(f"No significativo (p = {p_directo:.4f})")
            except Exception as e: st.error(f"Error en Wilcoxon: {e}")

st.sidebar.markdown("---")
st.sidebar.info("""
**Instrucciones Generales:**
1.  Selecciona el método.
2.  Ingresa los datos (números separados por comas).
3.  Haz clic en "Calcular...". Se mostrará un gráfico y los resultados.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Proyecto: Calculadora Estadística")