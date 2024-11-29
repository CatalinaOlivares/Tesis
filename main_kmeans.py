import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan

# Función para escalar los datos
def escalar_datos(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Función para aplicar K-Means clustering y agregar resultados al DataFrame
def aplicar_kmeans(X_scaled, df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    calcular_estadisticas_clusters(df, 'cluster')
    return df, kmeans

# Función para aplicar HDBSCAN y agregar resultados al DataFrame
def aplicar_hdbscan(X_scaled, df, min_cluster_size=5):
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    df['cluster_hdbscan'] = hdbscan_model.fit_predict(X_scaled)
    calcular_estadisticas_clusters(df, 'cluster_hdbscan')
    return df, hdbscan_model

# Función para calcular estadísticas de los clusters
def calcular_estadisticas_clusters(df, cluster_column):
    clusters = df[cluster_column].unique()
    print(f"\nEstadísticas para los clusters ({cluster_column}):")
    for cluster in clusters:
        if cluster == -1:  # Ignorar el ruido en HDBSCAN
            continue
        grupo = df[df[cluster_column] == cluster]
        tamano = len(grupo)
        promedio_tamano = grupo.shape[0] / len(clusters)  # Tamaño promedio del grupo
        desviacion_tamano = np.std(grupo.shape[0]) if tamano > 1 else 0
        desviacion_num_ciudades = np.std(grupo['num_nodos']) if tamano > 1 else 0
        print(f"Cluster {cluster}:")
        print(f"- Cantidad de instancias: {tamano}")
        print(f"- Tamaño promedio del grupo: {promedio_tamano:.2f}")
        print(f"- Desviación estándar del número de ciudades (num_nodos): {desviacion_num_ciudades:.2f}")

# Función para visualizar los clusters (compatible con K-Means y HDBSCAN)
def visualizar_clusters(df, X_scaled, model):
    plt.figure(figsize=(10, 6))
    
    # Verificar si es un modelo de HDBSCAN o K-Means
    if isinstance(model, hdbscan.HDBSCAN):
        title = f'HDBSCAN Clustering (min_cluster_size={model.min_cluster_size})'
        cluster_column = 'cluster_hdbscan'
    else:
        title = f'K-Means Clustering con {model.n_clusters} clusters'
        cluster_column = 'cluster'
    
    # Visualización de los clusters
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df[cluster_column],
                    palette='viridis', style=df[cluster_column], s=100)
    
    plt.title(title)
    plt.xlabel('Primera característica estandarizada')
    plt.ylabel('Segunda característica estandarizada')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()

# Función para aplicar PCA y mostrar la importancia de cada componente
def aplicar_pca(X_scaled, X, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Mostrar la varianza explicada por cada componente
    print(f'Varianza explicada por los primeros {n_components} componentes principales:')
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"Componente {i+1}: {var:.4f}")
    
    # Mostrar la importancia de las características en cada componente
    componentes = pd.DataFrame(pca.components_, columns=X.columns)
    print("\nImportancia de las características en cada componente:")
    print(componentes)
    # Identificar la característica con mayor peso en el primer componente
    max_peso_caracteristica = componentes.iloc[0].idxmax()
    print(f"\nLa característica que más pesa en el Componente 1 es: {max_peso_caracteristica}")
    return pca, X_pca

# Función para calcular métricas de similaridad (Silhouette y Davies-Bouldin) para cada grupo
def evaluar_clustering(X_scaled, labels, metodo='K-Means'):
    from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
    
    # Calcular el coeficiente de silueta global
    silhouette_avg = silhouette_score(X_scaled, labels)
    # Calcular el índice de Davies-Bouldin global
    davies_bouldin_avg = davies_bouldin_score(X_scaled, labels)
    
    print(f"\nMétricas de evaluación globales para {metodo}:")
    print(f"Coeficiente de Silueta: {silhouette_avg:.4f}")
    print(f"Índice de Davies-Bouldin: {davies_bouldin_avg:.4f}")
    
    # Calcular el coeficiente de silueta por grupo
    silueta_por_grupo = silhouette_samples(X_scaled, labels)
    for cluster in np.unique(labels):
        if cluster == -1:  # Ignorar el ruido en HDBSCAN
            continue
        silueta_cluster = silueta_por_grupo[labels == cluster].mean()
        print(f"Silueta promedio para el grupo {cluster}: {silueta_cluster:.4f}")

# Cargar el archivo CSV con las características
df = pd.read_csv('caracteristicas.csv')

# Eliminar espacios en los nombres de las columnas, si existen
df.columns = df.columns.str.strip()

# Eliminar las columnas innecesarias
columnas_a_eliminar = ['distancia_media', 'primer_cuartil_distancias','mediana_distancias','tercer_cuartil_distancias']
df = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns])

# Luego, elimina las columnas 'nombre_instancia' y 'num_nodos' para el clustering
X = df.drop(columns=['nombre_instancia', 'num_nodos']).copy()

# Verificar y manejar valores infinitos y NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Imputar NaN con la media de cada columna si hay valores faltantes
if X.isnull().values.any():
    X.fillna(X.mean(), inplace=True)

# Escalar los datos
X_scaled = escalar_datos(X)

# Aplicar PCA para identificar características problemáticas
n_components = min(len(X.columns), 10)  # Definir el número de componentes principales a analizar
pca, X_pca = aplicar_pca(X_scaled, X, n_components=n_components)

# Aplicar K-Means con el número de clusters deseado (por ejemplo, 10)
n_clusters = 5
df_clusterizado, kmeans_model = aplicar_kmeans(X_scaled, df, n_clusters=n_clusters)

# Mostrar el DataFrame con los clusters asignados por K-Means
print(df_clusterizado[['nombre_instancia', 'num_nodos', 'cluster']])

# Guardar los resultados de K-Means en un archivo CSV
df_clusterizado.to_csv('caracteristicas_normalizado.csv', index=False)
print("Clustering con K-Means completado y guardado en 'caracteristicas_normalizado.csv'.")

# Visualizar los clusters de K-Means
visualizar_clusters(df_clusterizado, X_scaled, kmeans_model)

# Evaluar el clustering de K-Means
print("Evaluación de K-Means:")
evaluar_clustering(X_scaled, df_clusterizado['cluster'], metodo='K-Means')

# Aplicar HDBSCAN
min_cluster_size =  2 # Ajusta este valor según tus necesidades
df_clusterizado_hdbscan, hdbscan_model = aplicar_hdbscan(X_scaled, df, min_cluster_size=min_cluster_size)

# Mostrar el DataFrame con los clusters asignados por HDBSCAN
print(df_clusterizado_hdbscan[['nombre_instancia', 'num_nodos', 'cluster_hdbscan']])

# Guardar los resultados de HDBSCAN en un archivo CSV
df_clusterizado_hdbscan.to_csv('caracteristicas_clusterizado_hdbscan.csv', index=False)
print("Clustering con HDBSCAN completado y guardado en 'caracteristicas_clusterizado_hdbscan.csv'.")

# Visualizar los clusters de HDBSCAN
visualizar_clusters(df_clusterizado_hdbscan, X_scaled, hdbscan_model)

# Evaluar el clustering de HDBSCAN
if (df_clusterizado_hdbscan['cluster_hdbscan'] != -1).sum() > 1:  # Verificar que haya suficientes puntos para evaluar
    print("Evaluación de HDBSCAN:")
    evaluar_clustering(X_scaled[df_clusterizado_hdbscan['cluster_hdbscan'] != -1], 
                       df_clusterizado_hdbscan['cluster_hdbscan'][df_clusterizado_hdbscan['cluster_hdbscan'] != -1], 
                       metodo='HDBSCAN')
else:
    print("No hay suficientes puntos en los clusters de HDBSCAN para calcular las métricas.")
