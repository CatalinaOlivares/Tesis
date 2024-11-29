import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis, mode

# Función para calcular métricas de la matriz de distancias
def calcular_metricas(distance_matrix, filename):
    # 1. Número de nodos (ciudades)
    num_nodos = distance_matrix.shape[0]

    # 2. Extraer distancias (sin ceros y sin duplicados)
    distancias = distance_matrix[np.triu_indices(num_nodos, k=1)]  # Solo distancias de la parte superior de la matriz

    # Asegurarse de que las distancias sean valores finitos
    distancias = distancias[np.isfinite(distancias)]

    # Verificar si hay suficientes distancias para calcular estadísticas
    if len(distancias) == 0:
        # Si no hay distancias válidas, asignar valores por defecto o saltar esta instancia
        return None

    # 3. Distancia media entre nodos
    distancia_media = np.mean(distancias)

    # 4. Varianza de las distancias entre nodos
    varianza_distancias = np.var(distancias)

    # 5. Simetría de las distancias
    simetria_distancias = np.allclose(distance_matrix, distance_matrix.T)

    # 6. Relación de las distancias mínima y máxima
    menor_arco = np.min(distancias)
    mayor_arco = np.max(distancias)
    relacion_min_max = menor_arco / mayor_arco if mayor_arco != 0 else 0

    # 7. Desviación de los vecinos más cercanos
    # Excluir distancias a sí mismo
    np.fill_diagonal(distance_matrix, np.inf)
    vecinos_cercanos = np.min(distance_matrix, axis=1)
    desviacion_vecinos_cercanos = np.std(vecinos_cercanos)

    # 8. Excentricidad media del grafo
    excentricidades = np.max(distance_matrix, axis=1)
    excentricidad_media = np.mean(excentricidades)

    # 9. Diámetro del grafo
    diametro = np.max(distancias)

    # 10. Asimetría de la matriz de distancias
    matriz_asimetrica = not simetria_distancias

    # 11. Mediana de las distancias
    mediana_distancias = np.median(distancias)

    # 12. Moda de las distancias
    moda_result = mode(distancias, nan_policy='omit')
    # Manejar el caso en que moda_result.mode es un escalar o un array
    if np.size(moda_result.mode) > 0:
        moda_distancias = moda_result.mode.item()
    else:
        moda_distancias = 0  # O asignar np.nan si lo prefieres

    # 13. Desviación estándar de las distancias
    desviacion_estandar_distancias = np.std(distancias)

    # 14. Valores máximo y mínimo de las distancias
    valor_maximo_distancias = mayor_arco
    valor_minimo_distancias = menor_arco

    # 15. Primer y tercer cuartil de las distancias
    primer_cuartil_distancias = np.percentile(distancias, 25)
    tercer_cuartil_distancias = np.percentile(distancias, 75)

    # 16. Coeficiente de variación de las distancias
    coeficiente_variacion_distancias = desviacion_estandar_distancias / distancia_media if distancia_media != 0 else 0

    # 17. Asimetría (Skewness) de las distancias
    skewness_distancias = skew(distancias, bias=False, nan_policy='omit')
    if np.isnan(skewness_distancias):
        skewness_distancias = 0  # O asignar np.nan

    # 18. Curtosis (Kurtosis) de las distancias
    kurtosis_distancias = kurtosis(distancias, bias=False, nan_policy='omit')
    if np.isnan(kurtosis_distancias):
        kurtosis_distancias = 0  # O asignar np.nan

    # 19. Entropía de las distancias
    # Crear un histograma con bins automáticos
    hist, bin_edges = np.histogram(distancias, bins='auto', density=True)
    # Calcular la probabilidad de cada bin
    prob_dist = hist / np.sum(hist)
    # Remover ceros para evitar log(0)
    prob_dist = prob_dist[prob_dist > 0]
    if len(prob_dist) > 0:
        entropia_distancias = -np.sum(prob_dist * np.log(prob_dist))
    else:
        entropia_distancias = 0  # O asignar np.nan

    # 20. Desviación media de las distancias
    desviacion_media_distancias = np.mean(np.abs(distancias - distancia_media))

    # 21. Desviación mediana de las distancias
    desviacion_mediana_distancias = np.median(np.abs(distancias - mediana_distancias))

    # Restaurar la diagonal original de la matriz de distancias
    np.fill_diagonal(distance_matrix, 0)

    # Guardar características en un diccionario
    caracteristicas = {
        'nombre_instancia': filename,
        'num_nodos': num_nodos,
        'distancia_media': distancia_media,
        'varianza_distancias': varianza_distancias,
        'simetria_distancias': simetria_distancias,
        'relacion_min_max': relacion_min_max,
        'desviacion_vecinos_cercanos': desviacion_vecinos_cercanos,
        #'excentricidad_media': excentricidad_media,
        'diametro': diametro,
        'matriz_asimetrica': matriz_asimetrica,
        'mediana_distancias': mediana_distancias,
        'moda_distancias': moda_distancias,
        'desviacion_estandar_distancias': desviacion_estandar_distancias,
        'valor_maximo_distancias': valor_maximo_distancias,
        'valor_minimo_distancias': valor_minimo_distancias,
        'primer_cuartil_distancias': primer_cuartil_distancias,
        'tercer_cuartil_distancias': tercer_cuartil_distancias,
        'coeficiente_variacion_distancias': coeficiente_variacion_distancias,
        'skewness_distancias': skewness_distancias,
        'kurtosis_distancias': kurtosis_distancias,
        'entropia_distancias': entropia_distancias,
        'desviacion_media_distancias': desviacion_media_distancias,
        'desviacion_mediana_distancias': desviacion_mediana_distancias
    }

    return caracteristicas

# Inicializar una lista para almacenar las características de todas las instancias
caracteristicas_todas = []

# Ruta de la carpeta donde están los archivos de matrices
carpeta_instancias = "./Matrix_2"

# Iterar sobre todos los archivos en la carpeta
for archivo in os.listdir(carpeta_instancias):
    # Cargar el archivo que contiene la matriz de distancias
    file_path = f'{carpeta_instancias}/{archivo}'

    # Leer el archivo que contiene la matriz de distancias
    distance_matrix = np.loadtxt(file_path)

    # Verificar si la matriz de distancias es válida
    if distance_matrix.size == 0:
        print(f"Advertencia: La instancia '{archivo}' tiene una matriz vacía. Se omitirá.")
        continue

    # Calcular las métricas para la matriz generada
    nombre_instancia = f'{archivo}'
    caracteristicas = calcular_metricas(distance_matrix, nombre_instancia)

    if caracteristicas is None:
        print(f"Advertencia: No se pudieron calcular características para '{archivo}'. Se omitirá.")
        continue

    # Añadir las características a la lista
    caracteristicas_todas.append(caracteristicas)

# Crear un DataFrame con todas las características
df_caracteristicas = pd.DataFrame(caracteristicas_todas)

# Guardar el DataFrame en un archivo CSV
df_caracteristicas.to_csv('caracteristicas_todas2.csv', index=False)

# Mostrar un mensaje de éxito
print("Todas las características se han guardado en 'caracteristicas.csv'.")


