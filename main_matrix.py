import tsplib95
import networkx as nx
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Función para calcular características de la matriz de distancias
def calcular_caracteristicas(distance_matrix):
    # Extraer distancias entre nodos
    distancias = np.array(distance_matrix[distance_matrix != 0]).flatten()

    # 1. Número de nodos (ciudades)
    num_nodos = distance_matrix.shape[0]

    # 2. Distancia media entre nodos
    distancia_media = np.mean(distancias)

    # 3. Varianza de las distancias entre nodos
    varianza_distancias = np.var(distancias)

    # 4. Simetría de las distancias
    simetria_distancias = np.allclose(distance_matrix, distance_matrix.T)

    # 5. Radio de la mínima esfera que contiene todas las ciudades (no disponible sin coordenadas)
    radio_minima_esfera = "No disponible sin coordenadas"

    # 6. Relación de las distancias mínima y máxima
    menor_arco = np.min(distancias)
    mayor_arco = np.max(distancias)
    relacion_min_max = menor_arco / mayor_arco

    # 7. Dimensionalidad del espacio (asumimos 2D, no disponible sin coordenadas)
    dimensionalidad = "No disponible sin coordenadas"

    # 8. Desviación estándar de las distancias a los vecinos más cercanos
    vecinos_cercanos = np.sort(distance_matrix, axis=1)[:, 1]  # Ignorar la primera columna (distancia consigo mismo)
    desviacion_vecinos_cercanos = np.std(vecinos_cercanos)

    # 9. Excentricidad media del grafo
    excentricidades = np.max(distance_matrix, axis=1)  # Excentricidad: distancia máxima desde cada nodo
    excentricidad_media = np.mean(excentricidades)

    # 10. Diámetro del grafo (máxima distancia entre nodos)
    diametro = np.max(distancias)

    # 11. Asimetría de la matriz de distancias
    matriz_asimetrica = not np.allclose(distance_matrix, distance_matrix.T)

    # Guardar características en un diccionario
    caracteristicas = {
        'num_nodos': num_nodos,
        'distancia_media': distancia_media,
        'varianza_distancias': varianza_distancias,
        'simetria_distancias': simetria_distancias,
        'radio_minima_esfera': radio_minima_esfera,
        'relacion_min_max': relacion_min_max,
        'dimensionalidad': dimensionalidad,
        'desviacion_vecinos_cercanos': desviacion_vecinos_cercanos,
        'excentricidad_media': excentricidad_media,
        'diametro': diametro,
        'matriz_asimetrica': matriz_asimetrica
    }

    return caracteristicas

# Directorios
carpeta_instancias = "./Instancias_diezmil"
carpeta_matrices = "./Matrix_1"

# Crear la carpeta Matrix si no existe
if not os.path.exists(carpeta_matrices):
    os.makedirs(carpeta_matrices)

# Inicializar una lista para almacenar las características de todas las instancias
caracteristicas_todas = []

# Iterar sobre todos los archivos en la carpeta
for archivo in os.listdir(carpeta_instancias):
    if archivo.endswith('.tsp'):
        print(f"Procesando: {archivo}")
        
        # Cargar el problema tsplib usando el archivo .tsp
        problem = tsplib95.load(os.path.join(carpeta_instancias, archivo))

        # Convertir el problema en un grafo de networkx
        graph = problem.get_graph()

        # Convertir el grafo en una matriz de distancias de numpy
        distance_matrix = nx.to_numpy_array(graph)

        # Guardar la matriz de distancias en un archivo .txt en la carpeta Matrix
        matrix_filename = os.path.join(carpeta_matrices, archivo.replace('.tsp', '.txt'))
        np.savetxt(matrix_filename, distance_matrix, fmt='%.2f')  # Guardar con formato de 4 decimales

        