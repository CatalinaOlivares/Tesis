import os
import time
import csv
import gc  # Importar el módulo de recolección de basura

# Función para procesar cada archivo (aquí puedes definir lo que quieres que haga con cada archivo)
def procesar_instancia(archivo_instancia):
    # Simulación de procesamiento (reemplaza esto con tu lógica)
    # print(f"Procesando instancia: {archivo_instancia}")
    time.sleep(1)  # Simula un proceso de 1 segundo
    return True

# Directorio de las instancias
carpeta_instancias = "./Instancias_menor_mil"

# Lista para almacenar los resultados
resultados = []

# Recorre cada archivo en la carpeta de instancias
for archivo_instancia in os.listdir(carpeta_instancias):
    ruta_archivo_instancia = os.path.join(carpeta_instancias, archivo_instancia)
    
    # Solo procesar si es un archivo
    if os.path.isfile(ruta_archivo_instancia):
        # Toma el tiempo de inicio
        tiempo_inicio = time.time()
        
        # Procesa la instancia
        procesar_instancia(ruta_archivo_instancia)
        
        # Toma el tiempo de fin
        tiempo_fin = time.time()
        
        # Calcula el tiempo de ejecución
        tiempo_ejecucion = tiempo_fin - tiempo_inicio
        
        # Almacena el nombre del archivo y el tiempo de ejecución
        resultados.append([archivo_instancia, tiempo_ejecucion])
        
        # Liberar memoria después de cada ciclo
        gc.collect()  # Fuerza la recolección de basura

# Guarda los resultados en un archivo CSV
with open('tiempos_ejecucion.csv', mode='w', newline='') as archivo_csv:
    escritor_csv = csv.writer(archivo_csv)
    # Escribe la cabecera del archivo CSV
    escritor_csv.writerow(['Nombre de la instancia', 'Tiempo de ejecución (segundos)'])
    
    # Escribe los datos de cada archivo procesado
    escritor_csv.writerows(resultados)

print("Tiempos de ejecución guardados en tiempos_ejecucion.csv")
