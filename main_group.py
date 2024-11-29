import os
import random

# Directorios de las carpetas
carpetas = ['Matrix_1', 'Matrix_2', 'Matrix_3']

# Leer todas las instancias de las carpetas
instancias = []
for carpeta in carpetas:
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.txt'):  # Cambia '.txt' por la extensión de tus archivos si es diferente
            instancias.append(os.path.join(carpeta, archivo))

# Verificar cuántas instancias tenemos en total
total_instancias = len(instancias)
print(f"Total de instancias: {total_instancias}")

# Crear 10 grupos vacíos
grupos = [[] for _ in range(10)]

# Asignar instancias a los grupos de forma aleatoria
for instancia in instancias:
    grupo_aleatorio = random.randint(0, 9)  # Selecciona un grupo aleatorio entre 0 y 9 (para 10 grupos)
    grupos[grupo_aleatorio].append(instancia)

# Mostrar el tamaño de cada grupo
for i, grupo in enumerate(grupos, 1):
    print(f"Grupo {i} tiene {len(grupo)} instancias")

# Mostrar las instancias de cada grupo
for i, grupo in enumerate(grupos, 1):
    print(f"\nGrupo {i}:")
    for instancia in grupo:
        print(instancia)
