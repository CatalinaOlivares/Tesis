import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from collections import defaultdict
import time
import copy
from itertools import product
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from stable_baselines3 import DQN
import numpy as np
from numba import njit
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete

#PARAMETROS
MAX_DEPTH = 6
import pytorch_lightning as pl
import torch
print(torch.cuda.is_available())
import torch.nn as nn
import pytorch_lightning as pl

class DQNLitModel(pl.LightningModule):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.network = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, self.env.action_space.n)
        )

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


##########################################################
################## CLASE INSTANCE ########################
##########################################################
class Instance:
    def __init__(self,
                 name,
                 optimal,
                 num_cities,
                 distance_matrix,
                 current_distance=None,
                 cities_to_visit=None,
                 cities_visited=None):
        # Atributos de la instancia
        self.name = name  # Nombre de la instancia
        self.optimal = optimal  # Distancia óptima conocida
        self.num_cities = num_cities  # Número de ciudades
        self.distance_matrix = np.array(distance_matrix)   # Matriz de distancias entre ciudades
        self.current_distance = current_distance  # Distancia actual (calculada)
        self.cities_to_visit = cities_to_visit  # Ciudades pendientes de visitar
        self.cities_visited = cities_visited  # Ciudades ya visitadas
        self.modified = False  # Atributo para rastrear modificaciones

    # Divide la instancia de 15,000 ciudades en subinstancias para cada agente
    def dividir_instancia(total_cities, num_agents):
        city_indices = np.arange(total_cities)
        np.random.shuffle(city_indices)  # Baraja las ciudades para una distribución aleatoria
        return np.array_split(city_indices, num_agents)

    def clone(self):
        # Crea una copia profunda de la instancia
        new_instance = Instance(
            self.name,
            self.optimal,
            self.num_cities,
            copy.deepcopy(self.distance_matrix),  # Copia profunda de la matriz de distancias
            self.current_distance,
            list(self.cities_to_visit),
            list(self.cities_visited)
        )
        return new_instance

    def get_current_distance(self):
        # Calcula la distancia de la ruta actual basada en las ciudades visitadas
        if len(self.cities_visited) <= 1:
            self.current_distance = 0
        else:
            total_cost = 0
            for i in range(len(self.cities_visited) - 1):
                from_city = self.cities_visited[i]
                to_city = self.cities_visited[i + 1]
                total_cost += self.distance_matrix[from_city][to_city]
            # Agrega el costo para regresar al punto de partida
            last_city = self.cities_visited[-1]
            first_city = self.cities_visited[0]
            total_cost += self.distance_matrix[last_city][first_city]
            self.current_distance = total_cost

    def get_other_distance(self, route):
        if len(route) <= 1:
            return 0
        else:
            route_np = np.array(route)
            total_cost = np.sum(self.distance_matrix[route_np[:-1], route_np[1:]])
            total_cost += self.distance_matrix[route_np[-1], route_np[0]]
            return total_cost



    def generate_initial_solution(self):
        # Genera una solución inicial aleatoria para el problema del viajante
        current_city = random.randint(0, self.num_cities - 1)  # Selecciona una ciudad de inicio al azar
        valid_path = [current_city]  # Comienza con una ruta que solo tiene la ciudad de inicio
        remaining_cities = set(range(self.num_cities)) - {current_city}  # Ciudades restantes para visitar
        total_cities_to_visit = self.num_cities  # Decide cuántas ciudades visitaremos en total
        while remaining_cities and len(valid_path) < total_cities_to_visit:
            # Encuentra ciudades conectadas a la ciudad actual y que no han sido visitadas
            next_cities = [city for city in remaining_cities if self.distance_matrix[current_city][city] > 0]
            if not next_cities:
                break
            next_city = random.choice(next_cities)  # Selecciona la siguiente ciudad al azar
            valid_path.append(next_city)
            remaining_cities.remove(next_city)
            current_city = next_city
        self.cities_visited = valid_path
        # Encuentra las ciudades que no han sido visitadas
        all_cities = list(range(self.num_cities))
        cities_to_visit = [city for city in all_cities if city not in valid_path]
        self.cities_to_visit = cities_to_visit
        return valid_path, cities_to_visit

    def choose_start_city_by_min_distance(self):
        # Suma de distancias para cada ciudad
        total_distances = np.sum(self.distance_matrix, axis=1)
        # Seleccionar la ciudad con la menor suma de distancias
        start_city = np.argmin(total_distances)
        return start_city 
    
    def generate_initial_solution_2(self):
        # Inicializa la ciudad de partida al azar
        current_city = self.choose_start_city_by_min_distance()
        valid_path = [current_city]  # Ruta válida comenzando con la ciudad inicial
        remaining_cities = set(range(self.num_cities)) - {current_city}  # Ciudades restantes por visitar

        # Mientras haya ciudades por visitar
        while remaining_cities:
            # Encuentra la ciudad más cercana a la ciudad actual
            next_city = min(remaining_cities, key=lambda city: self.distance_matrix[current_city][city])
            valid_path.append(next_city)  # Añade la ciudad más cercana a la ruta
            remaining_cities.remove(next_city)  # Elimina la ciudad de las restantes
            current_city = next_city  # Actualiza la ciudad actual

        self.cities_visited = valid_path  # Actualiza las ciudades visitadas
        self.cities_to_visit = []  # Todas las ciudades han sido visitadas

        # Retorna la solución inicial generada
        return valid_path, self.cities_to_visit

    def opt_swap(self,route,i,k):
      new_route = route[:i + 1]
      new_route.extend(reversed(route[i + 1: k + 1]))
      new_route.extend(route[k + 1:])
      return new_route

    #Métodos, terminales
    # Refinamiento
    def invert(self):
        prev_diff = abs(self.optimal- self.current_distance)
        if len(self.cities_visited) < 2:
            self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
            return False

        cities_visited_temp = list(self.cities_visited)
        initial_cost = self.get_other_distance(cities_visited_temp)
        previous_cost = initial_cost
        final_cost = initial_cost

        numIterations = len(cities_visited_temp) // 2

        if len(cities_visited_temp) % 2 == 0:
            centerNode = -1
        else:
            centerNode = len(cities_visited_temp) // 2

        for i in range(numIterations):
            if i != centerNode:
                temp_list = list(cities_visited_temp)  # Create a temporary copy of the list
                # Swap the elements
                temp_list[i], temp_list[-(i + 1)] = temp_list[-(i + 1)], temp_list[i]
                final_cost = self.get_other_distance(temp_list)

                if final_cost < previous_cost:
                    # Perform the swap on the original list
                    cities_visited_temp[i], cities_visited_temp[-(i + 1)] = cities_visited_temp[-(i + 1)], cities_visited_temp[i]
                    previous_cost = final_cost

        if previous_cost < initial_cost:
            self.cities_visited = list(cities_visited_temp)
            self.get_current_distance()
            self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
            return True
        else:
            self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
            return False

    def opt2(self):
      prev_diff = abs(self.optimal- self.current_distance)
      if len(self.cities_visited) < 4 or len(self.cities_to_visit) > 0:
          self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
          return False

      cities_visited_temp = list(self.cities_visited)
      distance_initial = self.get_other_distance(cities_visited_temp)
      current_distance = distance_initial
      c_i, c_k = -1, -1

      for i in range(len(cities_visited_temp)):
          for k in range(i + 2, len(cities_visited_temp)):
              i_next = 0 if i == len(cities_visited_temp) - 1 else i + 1
              k_next = 0 if k == len(cities_visited_temp) - 1 else k + 1
              gain_actual = self.distance_matrix[cities_visited_temp[i]][cities_visited_temp[i_next]] + \
                            self.distance_matrix[cities_visited_temp[k]][cities_visited_temp[k_next]]
              gain_candidato = self.distance_matrix[cities_visited_temp[i]][cities_visited_temp[k]] + \
                              self.distance_matrix[cities_visited_temp[i_next]][cities_visited_temp[k_next]]

              value = (distance_initial - gain_actual) + gain_candidato

              if value < current_distance:
                  current_distance = value
                  c_i = i
                  c_k = k

      if current_distance == distance_initial:
          self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
          return False
      else:
          new_route = self.opt_swap(cities_visited_temp, c_i, c_k)
          self.cities_visited = new_route
          self.get_current_distance()
          self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
          return True

    def opt2_2(self):
      prev_diff = abs(self.optimal- self.current_distance)
      if len(self.cities_visited) < 4 or len(self.cities_to_visit) > 0:
          self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
          return False

      cities_visited_temp = list(self.cities_visited)
      distance_initial = self.get_other_distance(cities_visited_temp)
      current_distance = distance_initial
      num_mejoras = 0

      for i in range(len(cities_visited_temp)):
          for k in range(i + 1, len(cities_visited_temp)):
              new_route = self.opt_swap(cities_visited_temp, i, k)
              new_distance = self.get_other_distance(new_route)

              if new_distance < current_distance:
                  cities_visited_temp[:] = new_route
                  current_distance = new_distance
                  num_mejoras += 1

      if (num_mejoras > 0):
        self.cities_visited = cities_visited_temp
        self.get_current_distance()
        self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
        return True
      else:
        self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
        return False

    def swap(self):
        prev_diff = abs(self.optimal - self.current_distance)
        if len(self.cities_visited) < 2 or len(self.cities_to_visit) > 0:
            self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
            return False

        # Convertimos cities_visited a un array de NumPy
        cities_visited_temp = np.array(self.cities_visited, dtype=np.int32)
        distance_initial = self.get_other_distance(cities_visited_temp)
        current_distance = distance_initial

        # Llamamos a la función optimizada con Numba
        c_i, c_j, current_distance = self.swap_optimized(cities_visited_temp, self.distance_matrix, current_distance)

        if current_distance == distance_initial:
            self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
            return False
        else:
            # Realizamos el intercambio final
            cities_visited_temp[c_i], cities_visited_temp[c_j] = cities_visited_temp[c_j], cities_visited_temp[c_i]
            self.cities_visited = cities_visited_temp.tolist()
            self.get_current_distance()
            self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
            return True

    @staticmethod
    @njit
    def swap_optimized(cities_visited_temp, distance_matrix, current_distance):
        c_i = -1
        c_j = -1
        n = len(cities_visited_temp)

        for i in range(n):
            for j in range(i + 1, n):
                # Intercambio temporal
                temp = cities_visited_temp[i]
                cities_visited_temp[i] = cities_visited_temp[j]
                cities_visited_temp[j] = temp

                # Calcular el costo de la ruta
                route_cost = 0.0
                for k in range(n - 1):
                    from_city = cities_visited_temp[k]
                    to_city = cities_visited_temp[k + 1]
                    route_cost += distance_matrix[from_city, to_city]
                # Agregar el costo de regresar a la ciudad inicial
                route_cost += distance_matrix[cities_visited_temp[-1], cities_visited_temp[0]]

                if route_cost < current_distance:
                    current_distance = route_cost
                    c_i = i
                    c_j = j

                # Revertir el intercambio
                cities_visited_temp[j] = cities_visited_temp[i]
                cities_visited_temp[i] = temp

        return c_i, c_j, current_distance

    def relocate(self):
      prev_diff = abs(self.optimal- self.current_distance)
      if len(self.cities_visited) < 3 or len(self.cities_to_visit) > 0:
          self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
          return False

      cities_visited_temp = list(self.cities_visited)
      distance_initial = self.get_other_distance(cities_visited_temp)
      current_distance = distance_initial
      c_i,c_j = -1, -1

      for i in range(len(cities_visited_temp)):
          for j in range(1, len(cities_visited_temp)):
              if i != j:
                  new_route = copy.copy(cities_visited_temp)
                  node_var = new_route.pop(i)
                  new_route.insert(j, node_var)

                  route_cost = self.get_other_distance(new_route)

                  if route_cost < current_distance:
                      current_distance = route_cost
                      c_i = i
                      c_j = j

      if current_distance == distance_initial:
          self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
          return False
      else:
          node_var = cities_visited_temp.pop(c_i)
          cities_visited_temp.insert(c_j, node_var)
          self.cities_visited = cities_visited_temp
          self.get_current_distance()
          self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
          return True

##########################################################
###################### CLASE NODE ########################
##########################################################
class Node:
    def __init__(self, type_, value=None, index=None):
            self.type = type_
            self.value = value
            self.index = index  # Agregar un atributo index
            self.left = None
            self.right = None
            self.parent = None  # Añadimos esta línea

    def execute(self, objetivo):
        if self.type == 'método':
            return getattr(objetivo, self.value)()
        elif self.type == 'función':
            left_result = self.left.execute(objetivo) if self.left else False
            if self.value == "if":
                if left_result:
                    return self.right.execute(objetivo) if self.right else False
                else:
                    return False
            elif self.value == "and":
                if not left_result:
                    return False
                return self.right.execute(objetivo) if self.right else False
            elif self.value == "or":
                if left_result:
                    return True
                return self.right.execute(objetivo) if self.right else False
            elif self.value == "while":
                # Límite de iteraciones para evitar bucles infinitos
                iterations_limit = len(objetivo.cities_visited)
                
                for _ in range(iterations_limit):
                    # Evaluar el nodo izquierdo (condición del while)
                    left_result = self.left.execute(objetivo) if self.left else False

                    # Si el nodo izquierdo devuelve False, terminar la ejecución del while
                    if not left_result:
                        return False

                    # Si el nodo izquierdo es verdadero, evaluar el nodo derecho (cuerpo del while)
                    right_result = self.right.execute(objetivo) if self.right else False

                    # Si el nodo derecho es verdadero, retornar True
                    if right_result:
                        return True

                # Si se ha alcanzado el límite de iteraciones sin que el nodo derecho devuelva True,
                # retornar False
                return False
        return False

    def to_string(self):
            if self.type == 'método':
                return self.value
            elif self.type == 'función':
                left_str = self.left.to_string() if self.left else ""
                right_str = self.right.to_string() if self.right else ""
                return f"{self.index}.{self.value}({left_str}, {right_str})"

    def depth(self):
            left_depth = self.left.depth() if self.left else 0
            right_depth = self.right.depth() if self.right else 0
            return 1 + max(left_depth, right_depth)

    def is_full(self):
            return self.left is not None and self.right is not None

    def current_depth(self):
        """Retorna la profundidad actual del nodo."""
        depth = 0
        temp = self
        while temp.parent:  # Mientras el nodo tenga un padre
            depth += 1
            temp = temp.parent  # Moverse al nodo padre
        return depth

    def get_current_node(self):
        """
        Devuelve el primer nodo de tipo None cuyo padre es de tipo 'función'.
        Si no existe tal nodo, retorna el nodo raíz si cumple con la condición de ser de tipo None.
        La búsqueda se realiza en orden de anchura (BFS).
        """
        
        # Si el nodo actual es el nodo raíz y cumple con las condiciones, devolver el nodo raíz
        if self.type is None and self.parent is None:
            return self

        # Cola para la búsqueda BFS
        queue = deque([self])

        while queue:
            current_node = queue.popleft()

            # Verificar si el nodo actual cumple con las condiciones
            if current_node.type is None and current_node.parent and current_node.parent.type == 'función':
                return current_node

            # Añadir los hijos a la cola de la BFS
            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)

        # Si no encontramos un nodo válido, devolvemos None
        return None
    
    def get_root_node(self):
            """Devuelve el nodo raíz del árbol dado cualquier nodo en el árbol."""
            current_node = self
            while current_node.parent:
                current_node = current_node.parent
            return current_node

    def modify_node_by_index(self, index, new_type, new_value):
        """
        Modifica el nodo con el índice dado, estableciendo un nuevo tipo y valor.
        
        :param index: El índice del nodo a modificar.
        :param new_type: El nuevo tipo a establecer ('función' o 'método').
        :param new_value: El nuevo valor a establecer para el nodo.
        :return: True si la modificación fue exitosa, False si el nodo con el índice dado no se encontró.
        """
        if self.index == index:
            self.type = new_type
            self.value = new_value
            return True
        # Buscar recursivamente en el hijo izquierdo
        if self.left and self.left.modify_node_by_index(index, new_type, new_value):
            return True
        # Buscar recursivamente en el hijo derecho
        if self.right and self.right.modify_node_by_index(index, new_type, new_value):
            return True
        # Si el nodo no se encontró en el subárbol actual
        return False
    
    def get_executable_tree(self):
        """
        Crea y retorna una versión del árbol que solo incluye los nodos con tipo definido (no None).
        
        :return: El nodo raíz del nuevo árbol ejecutable.
        """
        if self.type is None:
            return None
        
        executable_node = Node(self.type, self.value, self.index)
        
        if self.left:
            executable_node.left = self.left.get_executable_tree()
            if executable_node.left:
                executable_node.left.parent = executable_node
        
        if self.right:
            executable_node.right = self.right.get_executable_tree()
            if executable_node.right:
                executable_node.right.parent = executable_node
        
        return executable_node
    
    def clone(self):
            """Clona el nodo y todos sus descendientes."""
            cloned_node = Node(self.type, self.value)
            cloned_node.index = self.index
            if self.left:
                cloned_node.left = self.left.clone()
                cloned_node.left.parent = cloned_node

            if self.right:
                cloned_node.right = self.right.clone()
                cloned_node.right.parent = cloned_node

            return cloned_node
    
    def is_executable(self):
        """Verifica si todos los nodos hoja son de tipo 'método'."""
        if not self.left and not self.right:
            return self.type == 'método'
        left_executable = self.left.is_executable() if self.left else True
        right_executable = self.right.is_executable() if self.right else True
        return left_executable and right_executable
    
def create_empty_syntax_tree(max_depth = MAX_DEPTH, current_depth=0, index=0, parent=None):
    if current_depth > max_depth:
        return None

    # Crear el nodo actual con tipo y valor vacíos
    node = Node(type_=None, value=None, index=index)
    node.parent = parent
    
    if current_depth < max_depth:
        # Recursivamente crear los nodos hijos izquierdo y derecho
        node.left = create_empty_syntax_tree(max_depth, current_depth + 1, index=2*index + 1, parent=node)
        node.right = create_empty_syntax_tree(max_depth, current_depth + 1, index=2*index + 2, parent=node)

    return node
def calculate_erp(root, objetivos):
    distances = []
    erp_s_totals = []
    erp_s_sum = 0
    n_instances = len(objetivos)
    objetivos_temp = clonar_arreglo_instancias(objetivos)
    if root is None:
        for instance in objetivos_temp:
            erp_s_total = abs(instance.optimal - instance.current_distance) / abs(instance.optimal)
            erp_s_totals.append(erp_s_total)
            distances.append(instance.current_distance)
            erp_s_sum += erp_s_total
    else:
        root_executable = root.get_executable_tree()
        for objetivo in objetivos_temp:
            root_executable.execute(objetivo)
        for instance in objetivos_temp:
            erp_s_total = abs(instance.optimal - instance.current_distance) / abs(instance.optimal)
            erp_s_totals.append(erp_s_total)
            distances.append(instance.current_distance)
            erp_s_sum += erp_s_total
    erp_s_avg = erp_s_sum / n_instances
    return erp_s_avg, objetivos_temp, erp_s_totals, distances

def execute_root(root,objetivos):
    obj = clonar_arreglo_instancias(objetivos)
    root_executable = root.get_executable_tree()
    for objetivo in obj:
        root_executable.execute(objetivo)
    return obj
def get_tree_state_representation(root_node):
    if not root_node:
        return [0, 0, 0, 0, 0, 0, 0, 0,0,0]

    max_depth = 0
    function_counts = {
        'if': 0,
        'and': 0,
        'or': 0,
        'while': 0
    }
    method_counts = {
        'invert': 0,
        'opt2': 0,
        'opt2_2': 0,
        'swap': 0,
        'relocate': 0
    }
    queue = deque([(root_node, 0)])

    while queue:
        current_node, depth = queue.popleft()
        max_depth = max(max_depth, depth)

        if current_node.type == 'función':
            if current_node.value in function_counts:
                function_counts[current_node.value] += 1
        elif current_node.type == 'método':
            if current_node.value in method_counts:
                method_counts[current_node.value] += 1

        if current_node.left:
            queue.append((current_node.left, depth + 1))
        if current_node.right:
            queue.append((current_node.right, depth + 1))

    state_vector = [
        max_depth//2,
        function_counts['if'],
        function_counts['and'],
        function_counts['or'],
        function_counts['while'],
        method_counts['invert'],
        method_counts['opt2'],
        method_counts['opt2_2'],
        method_counts['swap'],
        method_counts['relocate']
    ]
    return state_vector   
def crear_arreglo_instancias(files):
  instancias = []
  for f in files:
    with open(f, 'r') as file:
      lines = file.readlines()
    # Procesamos las líneas
    n = int(lines[1].strip()) #Modificar segun instancias
    optimo = float(lines[0].strip()) #Modificar segun instancias
    matrix = [list(map(float, line.split())) for line in lines[2:]]
    instance = Instance(f,optimo,n,matrix)
    instance.generate_initial_solution_2()
    instance.get_current_distance()
    instancias.append(instance)
  return instancias
def crear_arreglo_instancias2(files):
  instancias = []
  for f in files:
    with open(f, 'r') as file:
      lines = file.readlines()
    # Procesamos las líneas
    n = int(lines[1].strip()) #Modificar segun instancias
    optimo = float(lines[0].strip()) #Modificar segun instancias
    matrix = [list(map(float, line.split())) for line in lines[2:]]
    instance = Instance(f,optimo,n,matrix)
    instance.generate_initial_solution()
    instance.get_current_distance()
    instancias.append(instance)
  return instancias
def leer_instancias():
    fileTrain = ["./prueba_instancias/att48.txt","./prueba_instancias/berlin52.txt","./prueba_instancias/eil51.txt","./prueba_instancias/pr76.txt","./prueba_instancias/st70.txt"]
    instancesTrain = crear_arreglo_instancias(fileTrain)
    return instancesTrain
def leer_instancias2():
    fileTrain = ["./prueba_instancias/att48.txt","./prueba_instancias/berlin52.txt","./prueba_instancias/eil51.txt","./prueba_instancias/pr76.txt","./prueba_instancias/st70.txt"]
    instancesTrain = crear_arreglo_instancias2(fileTrain)
    return instancesTrain
def clonar_arreglo_instancias(instancias):
    return [copy.copy(instance) for instance in instancias]

def calcular_erp_arreglo_instancias(instancias):
    distances = []
    erp_s_totals = []
    erp_s_sum = 0
    n_instances = len(instancias)
    objetivos_temp = clonar_arreglo_instancias(instancias)
    for instance in objetivos_temp:
            erp_s_total = abs(instance.optimal - instance.current_distance) / abs(instance.optimal)
            erp_s_totals.append(erp_s_total)
            distances.append(instance.current_distance)
            erp_s_sum += erp_s_total
    erp_s_avg = erp_s_sum / n_instances
    return erp_s_avg, objetivos_temp, erp_s_totals, distances



class TSPEnv(gym.Env):
    """Entorno personalizado para el problema TSP."""
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_instances, city_subset=None, seed=None):
        super(TSPEnv, self).__init__()

        # Almacena el subconjunto de ciudades para este agente
        self.city_subset = city_subset
        # Filtra las instancias iniciales para incluir solo las ciudades del subconjunto
        self.initial_instances = [instance for i, instance in enumerate(initial_instances) if i in city_subset] if city_subset else initial_instances

        self.nodo_actual = None
        self.estado_actual = None
        self.prev_erp_avg = None
        # Espacio de acciones
        self.action_space = Discrete(9)  # 9 acciones

        # Espacio de observaciones (vector de 10 elementos)
        self.observation_space = Box(low=0, high=(2**(MAX_DEPTH+1))-1, shape=(10*3,),dtype=np.int32)

        # Mapeo de acciones
        self._actions = {
            0: 'if',
            1: 'and',
            2: 'or',
            3: 'while',
            4: 'invert',
            5: 'opt2',
            6: 'opt2_2',
            7: 'swap',
            8: 'relocate'
        }
        self._actions2 = {v: k for k, v in self._actions.items()}
        self.max_steps_per_episode = (2**(MAX_DEPTH+1))-1
        self.steps_taken = 0
        # Variables para seguimiento
        self._used_actions = defaultdict(int)

        # Historial de estados (anterior anterior, anterior, actual)
        self.state_history = deque(maxlen=3)
        # Inicializamos el historial con tres estados vacíos (por ejemplo, arrays de ceros)
        empty_state = np.zeros((10,), dtype=np.int32)
        self.state_history.extend([empty_state, empty_state, empty_state])

    def reset(self, seed=None):
        # Reiniciar variables
        self._used_actions = defaultdict(int)
        self.current_instances = clonar_arreglo_instancias(self.initial_instances)
        self.prev_erp_avg, _, _, _ = calcular_erp_arreglo_instancias(self.current_instances)
        self.estado_actual = create_empty_syntax_tree()
        self.nodo_actual = self.estado_actual.get_current_node()
        # Limpiamos el historial de estados y lo inicializamos con estados vacíos
        empty_state = np.zeros((10,), dtype=np.int32)
        self.state_history = deque([empty_state, empty_state, empty_state], maxlen=3)

        # Agregamos el estado actual al historial
        self.state_history[-1] = get_tree_state_representation(self.estado_actual.get_executable_tree())

        info = self._get_info()
        obs = self._get_obs()
        return obs, info

    def step(self, action):
        done = False
        #print(self.estado_actual.to_string())
        #print(self.nodo_actual.index)
        # Obtener el nombre de la acción
        if isinstance(action, np.ndarray):
            action = action.item()
        action_name = self._actions[action]
        # Determinar el tipo de nodo
        if action_name in ("if", "and", "or", "while"):
            node_type = 'función'
        else:
            node_type = 'método'
        #print(action_name,node_type)

        # Obtener la profundidad actual del nodo
        current_depth = self.nodo_actual.current_depth()
        #print(current_depth)

        # Verificar si la acción es válida en el estado actual
        action_valid = True
        if current_depth == 0 and node_type != 'función':
            action_valid = False
        elif current_depth == MAX_DEPTH and node_type != 'método':
            # En los nodos hoja, solo se permiten métodos
            action_valid = False
        if action_valid:
            estado_siguiente, nodo_siguiente, done = self._do_action(self.estado_actual,self.nodo_actual,node_type,action_name)
            erp_avg_actual = self.prev_erp_avg
            # Calcular el ERP actual
            erp_avg_current, self.current_instances , _, _ = calculate_erp(estado_siguiente.get_executable_tree(),self.current_instances)
            # Calcular la recompensa
            #print(self.prev_erp_avg,erp_avg_current )
            reward = self._calculate_reward(self.prev_erp_avg, erp_avg_current, done)
            self.prev_erp_avg = erp_avg_current
            #print('Reward: ', reward)
            # Verificar si el episodio ha terminado
            if done:
                self.estado_actual = estado_siguiente.clone()
                self.nodo_actual = nodo_siguiente
            else:
                self.estado_actual = estado_siguiente.clone()
                self.nodo_actual = nodo_siguiente

        else:
            estado_siguiente, nodo_siguiente, done = self._do_action(self.estado_actual,self.nodo_actual,node_type,action_name)
            self.estado_actual = estado_siguiente.clone()
            self.nodo_actual = nodo_siguiente
            done = True
            reward = -1000

        # Actualizar el conteo de acciones usadas
        new_state = get_tree_state_representation(self.estado_actual.get_executable_tree())
        self.state_history.append(new_state)
        self._used_actions[action_name] += 1
        info = self._get_info()
        obs = self._get_obs()
        truncated = False
        return obs, reward, done,truncated, info

    def _calculate_reward(self, erp_avg_actual,erp_avg_siguiente, done):
            living_penalty = 0.05
            if (erp_avg_siguiente-erp_avg_actual) < 0:
                reward = (erp_avg_actual-erp_avg_siguiente) * 10
                if  0.04 <= erp_avg_siguiente <=  0.05:
                    reward += 50   
                elif 0.03 <= erp_avg_siguiente <  0.04:
                    reward += 100   
                elif 0.02 <= erp_avg_siguiente <  0.03:
                    reward += 500  
                    if done:
                        reward+= 50 
                elif 0.01 <= erp_avg_siguiente <  0.02:
                    reward += 1000  
                    if done:
                        reward+= 70 
                elif 0.0 <= erp_avg_siguiente < 0.01:
                    reward += 2000   
                    if done:
                        reward+= 100
            elif (erp_avg_siguiente-erp_avg_actual) == 0 and 0.04 <=erp_avg_siguiente <= 0.05 :
                reward =  10
            elif (erp_avg_siguiente-erp_avg_actual) == 0 and  0.03 <= erp_avg_siguiente < 0.04 :
                reward =  40
            elif (erp_avg_siguiente-erp_avg_actual) == 0 and  0.02 <= erp_avg_siguiente < 0.03 :
                reward =  60
            elif (erp_avg_siguiente-erp_avg_actual) == 0 and 0.01 <= erp_avg_siguiente < 0.02:
                reward = 100
            elif (erp_avg_siguiente-erp_avg_actual) == 0 and 0.0 <= erp_avg_siguiente < 0.01:
                reward = 200
            elif (erp_avg_siguiente-erp_avg_actual) == 0 and erp_avg_siguiente > 0.05:
                reward =  - living_penalty*2
            else:
                reward = - (erp_avg_siguiente-erp_avg_actual) * 2
            return reward

    def _do_action(self, estado_actual,nodo_actual,type, accion):
        estado_siguiente = estado_actual.clone()
        estado_siguiente.modify_node_by_index(nodo_actual.index,type,accion)
        if estado_siguiente.get_current_node() is not None:
                nodo_siguiente = estado_siguiente.get_current_node()
                done = False
        else:
                nodo_siguiente = nodo_actual.clone()
                done = True
        return estado_siguiente, nodo_siguiente, done

    def render(self, mode='human'):
        if self.estado_actual:
            print(self.estado_actual.get_root_node().to_string())

    def close(self):
        pass

    def _get_obs(self):
        observation = np.concatenate(list(self.state_history))
        arr = np.array(observation)
        return arr
    
    def _get_info(self):
        return {'data':self.estado_actual.to_string(), 'erp':self.prev_erp_avg, 'state':self._get_obs()}
    
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
import csv
import os 
#check_env(env) OK

class RewardErpCallback(BaseCallback,):
    def __init__(self, save_dir='.', verbose=0):
        super(RewardErpCallback, self).__init__(verbose)
        self.save_dir = save_dir
        self.episode_rewards = []
        self.epsilon_values = []
        self.episode_erps = []  # Añadimos esta lista para almacenar los ERPs
        self.episode_reward = 0.0
        self.episode_steps = 0
        csv_path = os.path.join(self.save_dir, 'training_metrics.csv')
        self.csv_file = open(csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file, delimiter=";")
        self.csv_writer.writerow(['Episode', 'Reward', 'ERP', 'Epsilon', 'Alg'])  # Añadimos 'ERP' al CSV

    def _on_step(self) -> bool:
        # Obtener la recompensa actual y el indicador done
        reward = self.locals['rewards']
        done = self.locals['dones']
        epsilon = self.model.exploration_rate
        self.epsilon_values.append(epsilon)
        # Acumular la recompensa
        self.episode_reward += reward[0]
        self.episode_steps += 1
        if done[0]:
            # Acceder al entorno desenvuelto
            # Acceder a prev_erp_avg
            #print(self.locals['infos'][0])
            erp_s_avg = self.locals['infos'][0]['erp']
            alg = self.locals['infos'][0]['data']
            #print(self.locals['infos'][0])
            # Registrar las métricas
            self.episode_rewards.append(self.episode_reward)
            self.episode_erps.append(erp_s_avg)  # Añadimos el ERP a la lista

            # Escribir en el archivo CSV
            #print([len(self.episode_rewards), self.episode_reward, erp_s_avg, epsilon, alg])
            self.csv_writer.writerow([len(self.episode_rewards), self.episode_reward, erp_s_avg, epsilon, alg])
            if self.verbose > 0:
                print(f"***Episode {len(self.episode_rewards)}: Reward = {self.episode_reward}, ERP = {erp_s_avg}***, epsilon = {epsilon}")
            # Reiniciar las variables
            self.episode_reward = 0.0
            self.episode_steps = 0
        return True

    def _on_training_end(self) -> None:
        # Cerrar el archivo CSV
        self.csv_file.close()

def log_to_file(filepath, message):
    with open(filepath, 'a') as file:
        file.write(message + '\n')

def evaluation_agent(env, model,current_time_seed):
            # Reset the environment to its initial state and get the initial observation (initial state)
            observation, info = env.reset(seed=current_time_seed)
            
            # Simulate the agent's actions for num_step time steps
            reward_acum = 0
            while True:
                # Choose a random action from the action space
                action, _states = model.predict(observation, deterministic=True)
                # Take the chosen action and observe the resulting state, reward, and termination status
                observation, reward, terminated, truncated, info = env.step(action)
                reward_acum += reward

                if terminated:
                    erp_s_avg, _, erp_s_totals, _ = calcular_erp_arreglo_instancias(env.current_instances)
                    return reward_acum, erp_s_avg,env.estado_actual.to_string(),erp_s_totals
                
if __name__ == '__main__':
    TIME_STEPS = 5000
    ITER = 50
    for iteration in range(ITER):
        current_time_seed = time.time()
        print(current_time_seed)
        
        iteration_folder = f"iteration_{iteration + 1}_vmc"
        os.makedirs(iteration_folder, exist_ok=True)
        iteration_log_file = os.path.join(iteration_folder, "iteration_log.txt")
        
        # Inicializa las instancias de entrenamiento
        initial_instances = leer_instancias2()
        env = TSPEnv(initial_instances)  # Crea el entorno
        
        # Usa el modelo DQN configurado con PyTorch Lightning
        model = DQNLitModel(env, learning_rate=0.0011)  # Pasa el entorno y el learning rate
        
        # Configura el entrenador de PyTorch Lightning para entrenar en GPU
        trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=ITER)
        # Registra el tiempo de inicio
        start_time = time.time()
        
        # Entrena el modelo
        trainer.fit(model)
        
        # Finaliza y registra el tiempo de ejecución
        end_time = time.time()
        execution_time = end_time - start_time
        log_to_file(iteration_log_file, f"Tiempo de ejecución: {execution_time:.2f} segundos")
        log_to_file(iteration_log_file, f"Semilla utilizada: {current_time_seed}")
        
        # Evalúa el modelo
        erp_s_avg, _, erp_s_totals, _ = calcular_erp_arreglo_instancias(initial_instances)
        log_to_file(iteration_log_file, f"ERP_AVG: {erp_s_avg} - ERP_INSTANCIA: {erp_s_totals}")
        
        # Guarda el modelo entrenado
        model.model.save(f"gaa_dqn_{iteration_folder}")
        
        # Ejecuta la evaluación del agente y registra resultados
        rewards, erps, algoritmo, erps_instancias = evaluation_agent(env, model.model, current_time_seed)
        log_to_file(iteration_log_file, f"ERP_AVG_EXEC: {erps} - ERP_INSTANCIA: {erps_instancias} - REWARD TOTAL: {rewards}")
        log_to_file(iteration_log_file, f"Algoritmo: {algoritmo}")
