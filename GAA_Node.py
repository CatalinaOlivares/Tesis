import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from collections import defaultdict
import time
import copy
from itertools import product
import pandas as pd

#PARAMETROS
MAX_DEPTH = 6

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
        self.distance_matrix = distance_matrix  # Matriz de distancias entre ciudades
        self.current_distance = current_distance  # Distancia actual (calculada)
        self.cities_to_visit = cities_to_visit  # Ciudades pendientes de visitar
        self.cities_visited = cities_visited  # Ciudades ya visitadas
        self.modified = False  # Atributo para rastrear modificaciones

    def clone(self):
        new_instance = Instance(
            self.name,
            self.optimal,
            self.num_cities,
            self.distance_matrix,  # No es necesario deepcopy
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
        # Calcula la distancia de cualquier ruta proporcionada
        if len(route) <= 1:
            return 0
        else:
            total_cost = 0
            for i in range(len(route) - 1):
                from_city = route[i]
                to_city = route[i + 1]
                total_cost += self.distance_matrix[from_city][to_city]
            # Agrega el costo para regresar al punto de partida
            last_city = route[-1]
            first_city = route[0]
            total_cost += self.distance_matrix[last_city][first_city]
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
   
    def generate_initial_solution_2(self):
        current_city = self.choose_start_city_by_min_distance()
        valid_path = [current_city]
        remaining_cities = set(range(self.num_cities)) - {current_city}

        while remaining_cities:
            if random.random() < 0.8:  # 80% elegir la ciudad más cercana
                next_city = min(remaining_cities, key=lambda city: self.distance_matrix[current_city][city])
            else:  # 20% elegir aleatoriamente
                next_city = random.choice(list(remaining_cities))
            valid_path.append(next_city)
            remaining_cities.remove(next_city)
            current_city = next_city

        self.cities_visited = valid_path
        self.cities_to_visit = []
        return valid_path, self.cities_to_visit


    def choose_start_city_by_min_distance(self):
        # Suma de distancias para cada ciudad
        total_distances = np.sum(self.distance_matrix, axis=1)
        # Seleccionar la ciudad con la menor suma de distancias
        start_city = np.argmin(total_distances)
        return start_city 
    '''
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
        return valid_path, self.cities_to_visit'''

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

    '''def opt2(self):
        prev_diff = abs(self.optimal - self.current_distance)
        if len(self.cities_visited) < 4 or len(self.cities_to_visit) > 0:
            self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
            return False

        cities_visited_temp = self.cities_visited[:]
        improved = True

        while improved:
            improved = False
            for i in range(len(cities_visited_temp) - 1):
                for k in range(i + 2, len(cities_visited_temp)):
                    if k == len(cities_visited_temp) - 1 and i == 0:
                        continue  # Evita romper la continuidad del tour
                    # Calcula el cambio en distancia si intercambiamos los bordes (i, i+1) y (k, k+1)
                    a, b = cities_visited_temp[i], cities_visited_temp[i + 1]
                    c, d = cities_visited_temp[k], cities_visited_temp[(k + 1) % len(cities_visited_temp)]

                    delta = (self.distance_matrix[a][c] + self.distance_matrix[b][d]) - \
                            (self.distance_matrix[a][b] + self.distance_matrix[c][d])

                    if delta < -1e-6:
                        # Realiza el intercambio
                        cities_visited_temp[i + 1:k + 1] = reversed(cities_visited_temp[i + 1:k + 1])
                        improved = True
                        break  # Mejora encontrada, reinicia la búsqueda desde el principio
                if improved:
                    break  # Reinicia el bucle externo

        new_distance = self.get_other_distance(cities_visited_temp)
        if new_distance < self.current_distance:
            self.cities_visited = cities_visited_temp
            self.current_distance = new_distance
            self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
            return True
        else:
            self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
            return False

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
            return True'''
    '''
    def opt2(self):
        prev_diff = abs(self.optimal - self.current_distance)
        if len(self.cities_visited) < 4 or len(self.cities_to_visit) > 0:
            self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
            return False

        cities_visited_temp = self.cities_visited[:]
        improved = True

        while improved:
            improved = False
            for i in range(len(cities_visited_temp) - 1):
                for k in range(i + 2, len(cities_visited_temp)):
                    if k == len(cities_visited_temp) - 1 and i == 0:
                        continue  # Evita romper la continuidad del tour
                    # Calcula el cambio en distancia si intercambiamos los bordes
                    a, b = cities_visited_temp[i], cities_visited_temp[i + 1]
                    c, d = cities_visited_temp[k], cities_visited_temp[(k + 1) % len(cities_visited_temp)]

                    delta = (self.distance_matrix[a][c] + self.distance_matrix[b][d]) - \
                            (self.distance_matrix[a][b] + self.distance_matrix[c][d])

                    if delta < -1e-6:
                        cities_visited_temp[i + 1:k + 1] = reversed(cities_visited_temp[i + 1:k + 1])
                        improved = True
                        break
                if improved:
                    break

        new_distance = self.get_other_distance(cities_visited_temp)
        if new_distance < self.current_distance:
            self.cities_visited = cities_visited_temp
            self.current_distance = new_distance
            self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
            return True
        else:
            self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
            return False

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
        return False'''

    def swap(self):
      prev_diff = abs(self.optimal- self.current_distance)
      if len(self.cities_visited) < 2 or len(self.cities_to_visit) > 0:
          self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
          return False

      cities_visited_temp = list(self.cities_visited)
      distance_initial = self.get_other_distance(cities_visited_temp)
      current_distance = distance_initial
      c_i = -1
      c_j = -1

      for i in range(len(cities_visited_temp)):
          for j in range(i + 1, len(cities_visited_temp)):
              new_route = copy.copy(cities_visited_temp)

              node_var = new_route[i]
              new_route[i] = new_route[j]
              new_route[j] = node_var

              route_cost = self.get_other_distance(new_route)

              if route_cost < current_distance:
                  current_distance = route_cost
                  c_i = i
                  c_j = j

      if current_distance == distance_initial:
          self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
          return False
      else:
          node_var = cities_visited_temp[c_i]
          cities_visited_temp[c_i] = cities_visited_temp[c_j]
          cities_visited_temp[c_j] = node_var
          self.cities_visited = cities_visited_temp
          self.get_current_distance()
          self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
          return True

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
    def replace_function(self):
        """Reemplaza la función del nodo actual con otra función aleatoria."""
        if self.type == 'función':
            allowed_functions = ['if', 'and', 'or', 'while']
            new_function = random.choice([f for f in allowed_functions if f != self.value])
            print(f"Función {self.value} reemplazada por: {new_function}")
            self.value = new_function

    def replace_terminal(self):
        """Reemplaza un nodo terminal por otro terminal aleatorio., 'opt2', 'opt2_2'"""
        if not self.left and not self.right and self.type == 'método':
            allowed_terminals = ['invert', 'swap', 'relocate']
            new_terminal = random.choice([t for t in allowed_terminals if t != self.value])
            print(f"Nodo terminal reemplazado por: {new_terminal}")
            self.value = new_terminal
        else:
            if self.left:
                self.left.replace_terminal()
            if self.right:
                self.right.replace_terminal()

    def shrink_tree(self):
        """Reduce el árbol al nodo terminal más cercano."""
        if self.type == 'función' and (self.left or self.right):
            # Escoge un subárbol aleatorio y lo convierte en el árbol principal
            if self.left and self.right:
                subtree = random.choice([self.left, self.right])
            elif self.left:
                subtree = self.left
            else:
                subtree = self.right

            self.type = subtree.type
            self.value = subtree.value
            self.left = subtree.left
            self.right = subtree.right
            print(f"Árbol reducido al nodo: {self.value}")

    def add_random_subtree(self, max_depth=3):
        """Agrega un subárbol aleatorio al nodo actual si es terminal o un subárbol vacío."""
        if not self.left and not self.right:
            allowed_functions = ['if', 'and', 'or', 'while']
            self.type = 'función'
            self.value = random.choice(allowed_functions)
            print(f"Nodo terminal convertido en función: {self.value}")
            # Crear subárboles con profundidad aleatoria, controlada por max_depth
            self.left = create_syntax_tree(max_depth - 1, current_depth=1, parent=self)
            self.right = create_syntax_tree(max_depth - 1, current_depth=1, parent=self)
        elif self.left and self.right:
            # Recursivamente intenta agregar subárbol en uno de los hijos si ya tiene un tipo asignado
            if random.choice([True, False]):
                self.left.add_random_subtree(max_depth - 1)
            else:
                self.right.add_random_subtree(max_depth - 1)


    def swap_subtrees(self):

        """Intercambia los subárboles izquierdo y derecho del nodo actual si ambos existen."""
        if self.left and self.right:
            # Verifica que ambos subárboles sean válidos antes de intercambiarlos
            self.left, self.right = self.right, self.left
            print("Subárboles intercambiados.")
        else:
            # Si alguno de los subárboles es None, intenta crear uno aleatorio para permitir el intercambio
            if not self.left:
                self.left = create_syntax_tree(2, current_depth=1, parent=self)
            if not self.right:
                self.right = create_syntax_tree(2, current_depth=1, parent=self)
            print("Subárboles creados para el intercambio y luego intercambiados.")


    def mutate_subtree(self, depth, mutation_probability=0.5):
        """Aplica mutación al subárbol de este nodo hasta una profundidad específica."""
        if depth > 0 and random.random() < mutation_probability:
            # Mutación aleatoria en el hijo izquierdo o derecho
            if random.choice([True, False]) and self.left:
                self.left.mutate_subtree(depth - 1, mutation_probability)
            elif self.right:
                self.right.mutate_subtree(depth - 1, mutation_probability)
        else:
            # Mutación en el nodo actual si es una función o terminal
            if self.type == 'función':
                self.replace_function()
            elif self.type == 'método':
                self.replace_terminal()

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
    '''
    def get_executable_tree(self):
        """
        Crea y retorna una versión del árbol que solo incluye los nodos con tipo definido (no None).
        
        :return: El nodo raíz del nuevo árbol ejecutable o None si el árbol no es ejecutable.
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
        
        # Validación: verificar que ambos hijos tengan tipo asignado en nodos de función
        if executable_node.type == 'función' and (executable_node.left is None or executable_node.right is None):
            return None  # Retorna None si no es ejecutable
        
        return executable_node'''
    def get_executable_tree(self):
        """
        Crea y retorna una versión del árbol que solo incluye los nodos con tipo definido (no None).
        Si un nodo no es válido, se reemplaza por un nodo válido predeterminado.
        """
        # Si el nodo actual no tiene un tipo definido, crea un nodo predeterminado
        if self.type is None:
            return Node(type_='método', value='invert', index=self.index)  # Nodo predeterminado

        # Crea una copia del nodo actual
        executable_node = Node(self.type, self.value, self.index)

        # Procesa el subárbol izquierdo
        if self.left:
            executable_node.left = self.left.get_executable_tree()
            if executable_node.left:
                executable_node.left.parent = executable_node
        else:
            # Si falta el subárbol izquierdo y es necesario, crea un nodo predeterminado
            if self.type == 'función':
                executable_node.left = Node(type_='método', value='invert', index=2 * self.index + 1)
                executable_node.left.parent = executable_node

        # Procesa el subárbol derecho
        if self.right:
            executable_node.right = self.right.get_executable_tree()
            if executable_node.right:
                executable_node.right.parent = executable_node
        else:
            # Si falta el subárbol derecho y es necesario, crea un nodo predeterminado
            if self.type == 'función':
                executable_node.right = Node(type_='método', value='invert', index=2 * self.index + 2)
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
    
#-----------------------------------------
import random
from collections import deque

def select_random_node(root_node):
    """
    Selecciona un nodo de forma aleatoria en el árbol, basado en su índice.
    
    :param root_node: Nodo raíz del árbol de sintaxis.
    :return: Nodo seleccionado aleatoriamente o None si el árbol está vacío.
    """
    if not root_node:
        return None  # Si el árbol está vacío, retorna None

    # Recorrido BFS para recolectar todos los nodos
    queue = deque([root_node])
    nodes = []

    while queue:
        current_node = queue.popleft()
        nodes.append(current_node)  # Agrega el nodo actual a la lista

        # Agrega los hijos del nodo actual a la cola
        if current_node.left:
            queue.append(current_node.left)
        if current_node.right:
            queue.append(current_node.right)

    # Selecciona un nodo aleatorio de la lista
    selected_node = random.choice(nodes)
    return selected_node

    
    #AGREGO VALIDACION DE NO CREAR HIJOS EN LOS TERMINALES
def create_syntax_tree(max_depth=MAX_DEPTH, current_depth=0, index=0, parent=None):
    # Detener si hemos alcanzado la profundidad máxima
    if current_depth > max_depth:
        return None

    # Decidir si el nodo será un terminal o una función
    if current_depth == max_depth or (current_depth > 0 and random.choice([True, False])):
        # Crear un nodo terminal
        node_type = 'método'
        node_value = random.choice(['invert', 'swap', 'relocate'])
    else:
        # Crear un nodo de función
        node_type = 'función'
        node_value = random.choice(['if', 'and', 'or', 'while'])

    # Crear el nodo actual con tipo y valor asignados
    node = Node(type_=node_type, value=node_value, index=index)
    node.parent = parent

    # Solo crear hijos si el nodo es de tipo función
    if node_type == 'función':
        node.left = create_syntax_tree(max_depth, current_depth + 1, index=2 * index + 1, parent=node)
        node.right = create_syntax_tree(max_depth, current_depth + 1, index=2 * index + 2, parent=node)

    return node


def prune_tree(node):
    if not node:
        return None
    if node.type == 'función' and (not node.left or not node.right):
        return None  # Elimina nodos función sin hijos válidos
    node.left = prune_tree(node.left)
    node.right = prune_tree(node.right)
    return node

'''
def aplicate_mutation(root_node, mutation_index):
    """
    Aplica una mutación específica al árbol de sintaxis basado en el índice de mutación,
    validando previamente si la mutación es posible según las características del árbol.
    
    :param root_node: Nodo raíz del árbol de sintaxis.
    :param mutation_index: Índice de la mutación a aplicar.
    :return: El nodo raíz del árbol modificado si la mutación es válida, de lo contrario, retorna el árbol original.
    """
    if mutation_index == 0:
        # Mutación: replace_terminal
        if root_node.is_executable():
            print("Aplicando mutación: replace_terminal")
            root_node.replace_terminal()
        else:
            print("Mutación replace_terminal no es válida en el estado actual del árbol.")
            
    elif mutation_index == 1:
        # Mutación: shrink_tree
        if root_node.depth() > 1:
            print("Aplicando mutación: shrink_tree")
            root_node.shrink_tree()
        else:
            print("Mutación shrink_tree no es válida porque la profundidad del árbol es 1 o menor.")
            
    elif mutation_index == 2:
        # Mutación: add_random_subtree
        if root_node.is_full():
            print("Aplicando mutación: add_random_subtree")
            root_node.add_random_subtree(max_depth=3)
        else:
            print("Mutación add_random_subtree no es válida, ya que el nodo actual no permite añadir un subárbol.")
            
    elif mutation_index == 3:
        # Mutación: swap_subtrees
        if root_node.left and root_node.right:
            print("Aplicando mutación: swap_subtrees")
            root_node.swap_subtrees()
        else:
            print("Mutación swap_subtrees no es válida, ya que no hay subárboles para intercambiar.")
            
    elif mutation_index == 4:
        # Mutación: mutate_subtree
        if root_node.depth() > 1:
            print("Aplicando mutación: mutate_subtree")
            root_node.mutate_subtree(depth=2, mutation_probability=0.5)
        else:
            print("Mutación mutate_subtree no es válida porque la profundidad del árbol es insuficiente.")
    else:
        print(f"Índice de mutación no válido: {mutation_index}")
    
    return root_node'''
    

#DEFINIR ACCIONES como metodos
def aplicate_mutation(root_node, mutation_index):
    """
    Aplica una mutación específica al árbol de sintaxis basado en el índice de mutación,
    validando previamente si la mutación es posible según las características del árbol.
    
    :param root_node: Nodo raíz del árbol de sintaxis.
    :param mutation_index: Índice de la mutación a aplicar.
    :return: El nodo raíz del árbol modificado si la mutación es válida, de lo contrario, retorna el árbol original.
    """
    allowed_actions = allowed_function(root_node)
    if random.random() < 0.2 and len(allowed_actions) > 1:
        mutation_index = random.choice([action for action in allowed_actions if action != mutation_index])

    if mutation_index == 0:
        # Mutación: replace_terminal
        if root_node.is_executable():
            print("Aplicando mutación: replace_terminal")
            root_node.replace_terminal()
        else:
            print("Mutación replace_terminal no es válida en el estado actual del árbol.")
            
    elif mutation_index == 1:
        # Mutación: shrink_tree
        if root_node.depth() > 1:
            print("Aplicando mutación: shrink_tree")
            root_node.shrink_tree()
        else:
            print("Mutación shrink_tree no es válida porque la profundidad del árbol es 1 o menor.")
            
    elif mutation_index == 2:
        # Mutación: add_random_subtree
        if root_node.is_full():
            print("Aplicando mutación: add_random_subtree")
            root_node.add_random_subtree(max_depth=3)
        else:
            print("Mutación add_random_subtree no es válida, ya que el nodo actual no permite añadir un subárbol.")
            
    elif mutation_index == 3:
        # Mutación: swap_subtrees
        if root_node.left and root_node.right:
            print("Aplicando mutación: swap_subtrees")
            root_node.swap_subtrees()
        else:
            print("Mutación swap_subtrees no es válida, ya que no hay subárboles para intercambiar.")
            
    elif mutation_index == 4:
        # Mutación: mutate_subtree
        if root_node.depth() > 1:
            print("Aplicando mutación: mutate_subtree")
            root_node.mutate_subtree(depth=2, mutation_probability=0.5)
        else:
            print("Mutación mutate_subtree no es válida porque la profundidad del árbol es insuficiente.")
    else:
        print(f"Índice de mutación no válido: {mutation_index}")
    
    return root_node


#veo el timepo
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

'''def leer_instancias2():
    fileTrain = ["./prueba_instancias/att48.txt","./prueba_instancias/berlin52.txt","./prueba_instancias/eil51.txt","./prueba_instancias/pr76.txt","./prueba_instancias/st70.txt"]
    instancesTrain = crear_arreglo_instancias2(fileTrain)
    return instancesTrain'''
def leer_instancias2():
    fileTrain = ["./prueba_instancias/att48.txt"]
    instancesTrain = crear_arreglo_instancias2(fileTrain)
    return instancesTrain

def clonar_arreglo_instancias(instancias):
  instances_temp = []
  for instance in instancias:
    instance_temp =  instance.clone()
    instances_temp.append(instance_temp)
  return instances_temp
  
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



import random

def get_random_root_node(max_depth=MAX_DEPTH, current_depth=0, index=0, parent=None):
    """
    Genera un árbol de sintaxis aleatorio, asegurando que los nodos de tipo 'método' no tengan hijos.
    Retorna el nodo raíz.
    """
    if current_depth > max_depth:
        return None  # Detener si hemos alcanzado la profundidad máxima

    # Decide el tipo de nodo: 'función' o 'método'
    node_type = 'función' if current_depth < max_depth and random.choice([True, False]) else 'método'
    node_value = random.choice(['if', 'and', 'or', 'while']) if node_type == 'función' else random.choice(['invert', 'swap', 'relocate'])

    # Crear el nodo actual con tipo y valor asignados
    node = Node(type_=node_type, value=node_value, index=index)
    node.parent = parent
    
    if node_type == 'función':
        # Solo crear hijos si es un nodo de tipo 'función'
        node.left = get_random_root_node(max_depth, current_depth + 1, index=2*index + 1, parent=node)
        node.right = get_random_root_node(max_depth, current_depth + 1, index=2*index + 2, parent=node)

    return node

# Nueva función de representación de estado
def get_tree_state_representation(root_node):
    if not root_node:
        return [0] * 11  # Vector de tamaño fijo con ceros si el árbol está vacío

    # Inicializar contadores y variables
    max_depth = 0
    function_counts = defaultdict(int)
    method_counts = defaultdict(int)
    total_nodes = 0
    current_node_index = 0
    current_node_depth = 0

    # BFS para recorrer el árbol
    queue = deque([(root_node, 0)])  # (nodo, profundidad)
    while queue:
        current_node, depth = queue.popleft()
        total_nodes += 1
        max_depth = max(max_depth, depth)

        # Conteos de funciones y métodos
        if current_node.type == 'función':
            function_counts[current_node.value] += 1
        elif current_node.type == 'método':
            method_counts[current_node.value] += 1

        # Actualizar índice y profundidad del nodo actual
        if current_node.index > current_node_index:
            current_node_index = current_node.index
            current_node_depth = depth

        # Añadir hijos a la cola
        if current_node.left:
            queue.append((current_node.left, depth + 1))
        if current_node.right:
            queue.append((current_node.right, depth + 1))

    # Crear el vector de estado
    state_vector = [
        max_depth,                                # Profundidad máxima del árbol
        total_nodes,                              # Total de nodos
        function_counts['if'],                   # Conteo de funciones 'if'
        function_counts['and'],                  # Conteo de funciones 'and'
        function_counts['or'],                   # Conteo de funciones 'or'
        function_counts['while'],                # Conteo de funciones 'while'
        method_counts['invert'],                 # Conteo de métodos 'invert'
        method_counts['swap'],                   # Conteo de métodos 'swap'
        method_counts['relocate'],               # Conteo de métodos 'relocate'
        current_node_index,                      # Índice del nodo actual
        current_node_depth                       # Profundidad del nodo actual
    ]

    return state_vector




def allowed_function(root_node):
        """
        Determina qué acciones están permitidas para el nodo raíz actual.
        
        :param root_node: Nodo raíz del árbol de sintaxis.
        :return: Lista de índices de acciones permitidas.
        """
        allowed_actions = []
        
        # Verifica si el nodo es "ejecutable" (tiene todas las condiciones para ejecutar replace_terminal)
        if root_node.is_executable():
            allowed_actions.append(0)  # Acción: replace_terminal
        
        # Verifica si se puede aplicar shrink_tree (requiere que la profundidad sea mayor que 1)
        if root_node.depth() > 1:
            allowed_actions.append(1)  # Acción: shrink_tree
        
        # Verifica si se puede aplicar add_random_subtree (requiere que el nodo esté completo)
        if root_node.is_full():
            allowed_actions.append(2)  # Acción: add_random_subtree
        
        # Verifica si se puede aplicar swap_subtrees (requiere que el nodo tenga ambos subárboles)
        if root_node.left and root_node.right:
            allowed_actions.append(3)  # Acción: swap_subtrees
        
        # Verifica si se puede aplicar mutate_subtree (requiere una profundidad mayor que 1)
        if root_node.depth() > 1:
            allowed_actions.append(4)  # Acción: mutate_subtree
        
        # Validación adicional para una acción de reemplazo de función (acción 5, si es aplicable)
        #if root_node.type == 'función':
           # allowed_actions.append(5)  # Acción: replace_function
        if not allowed_actions:
            allowed_actions.append(1)  # Default to shrink_tree if no action is allowed

        return allowed_actions

def calculate_reward(delta, optimal_distance):
        """
        Calcula la recompensa basada en el delta entre la solución actual y la óptima.

        :param delta: Diferencia entre la distancia óptima y la actual (optimal - current).
        :param optimal_distance: Distancia óptima para normalizar las bonificaciones.
        :return: Recompensa calculada.
        """
        # Base de recompensa
        reward = 0

        if delta > 0:
            # Recompensa proporcional al delta (mayor delta, mayor recompensa)
            reward += 100 * delta  # Ajusta la escala según lo que funcione mejor

            # Bonus adicional si el delta es menor a ciertos umbrales
            if delta <= 0.03 * optimal_distance:
                reward += 50  # Bonus menor si está dentro del 3% de la distancia óptima
            if delta <= 0.01 * optimal_distance:
                reward += 1000  # Bonus mayor si está dentro del 1% de la distancia óptima

        else:
            # Penalización proporcional al valor absoluto del delta
            reward -= 100 * abs(delta)  # Ajusta la penalización según sea necesario

            # Bonus por mejorar significativamente el ERP
        if delta > 0.05 * optimal_distance:
            reward += 200
        return reward

##########################################################
###################### CLASE ENV ########################
##########################################################
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete

class TSPEnv(gym.Env):
    """Entorno personalizado para el problema TSP."""
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_instances, seed= None):
            super(TSPEnv, self).__init__()
            # Instancias iniciales del TSP
            self.initial_instances = initial_instances
            self.current_root = create_syntax_tree()  # Crear el árbol una sola vez
            self.estado_actual = None
            self.current_node = None  # Nodo actual seleccionado
            self.prev_erp_avg, _, _, _ = calculate_erp(
            self.current_root,
            clonar_arreglo_instancias(self.initial_instances)
        )
            # Espacio de acciones
            self.action_space = Discrete(6)  #  acciones

            # Espacio de observaciones (vector de 10 elementos)
            #self.observation_space = Box(low=0, high=(2**(MAX_DEPTH+1))-1, shape=(10*2,),dtype=np.int32)
            # Espacio de observaciones basado en el nuevo vector de estado
            self.observation_space = Box(low=0, high=100, shape=(11,), dtype=np.float32)
            #self.state_history = deque(maxlen=2)
            
            # Mapeo de acciones
            self._actions = {
                0: 'replace_terminal',
                1: 'shrink_tree',
                2: 'add_random_subtree',
                3: 'swap_subtrees',
                4: 'mutate_subtree',
                #5: 'replace_function'
            }

            # Inversa del mapeo de acciones
            self._actions2 = {v: k for k, v in self._actions.items()}
            self.max_steps_per_episode = 1
            self.steps_taken = 0

            # Historial de estados (anterior anterior, anterior, actual)
            self.state_history = deque(maxlen=2)
            # Inicializamos el historial con 2 estados vacíos (por ejemplo, arrays de ceros)
            empty_state = np.zeros((11,), dtype=np.int32)
            self.state_history.extend([empty_state, empty_state])

    def _get_obs(self):
        # Obtén el estado actual del árbol
        current_state = get_tree_state_representation(self.current_root)
        return np.array(current_state, dtype=np.float32)
    
    def _get_info(self):
        #return {'data':self.current_root.to_string(), 'erp':self.prev_erp_avg, 'state':self._get_obs()}
        return {
            'tree_structure': self.current_root.to_string(),
            'erp': self.prev_erp_avg,
            'state': self._get_obs()
        }
    
    def count_functions(self, root_node):
        """
        Cuenta el número de nodos de tipo 'función' en el árbol de nodos.
        """
        if not root_node:
            return 0
        
        # Contar el nodo actual si es de tipo 'función'
        count = 1 if root_node.type == 'función' else 0

        # Recursión para contar los nodos 'función' en los subárboles izquierdo y derecho
        if root_node.left:
            count += self.count_functions(root_node.left)
        if root_node.right:
            count += self.count_functions(root_node.right)

        return count
    
    def reset(self, seed=None):
        # Reiniciar variables
        #self.current_root = create_syntax_tree()  # Asume que esta función crea un árbol de sintaxis inicial------------------------------------------------------------------------->ver
        self.prev_erp_avg, _, _, _ = calculate_erp(self.current_root, clonar_arreglo_instancias(self.initial_instances))
        # Actualiza el historial de estados
        initial_state = get_tree_state_representation(self.current_root)
        self.state_history = deque([np.zeros_like(initial_state), initial_state], maxlen=2)

        return self._get_obs(), self._get_info()
    
    

    
    def _evaluate_erp_change(self, erp_anterior, erp_posterior):
        delta = erp_anterior - erp_posterior  # Cambio en el ERP
        reward = calculate_reward(delta, erp_anterior)

        # Penalización adicional por tamaño del árbol (opcional)
        num_nodes = len(get_tree_state_representation(self.current_root))
        size_penalty = num_nodes * 0.002
        reward -= size_penalty
        return reward

    '''def step(self, action):
    """
    Realiza un paso en el entorno aplicando la acción seleccionada.
    """
    allowed_actions = allowed_function(self.current_root)
    if action in allowed_actions:
        new_root = aplicate_mutation(self.current_root, action)
        self.current_root = prune_tree(new_root)  # Asegúrate de que el árbol sea válido
        executable_tree = self.current_root.get_executable_tree()
        
        if executable_tree:  # Si el árbol es ejecutable, calcula ERP
            new_erp_avg, _, _, _ = calculate_erp(executable_tree, clonar_arreglo_instancias(self.initial_instances))
        else:
            new_erp_avg = self.prev_erp_avg  # No hay cambio si el árbol no es ejecutable

        reward = self._evaluate_erp_change(self.prev_erp_avg, new_erp_avg)
        self.prev_erp_avg = new_erp_avg
        self.current_root = new_root.clone()

        obs = self._get_obs()
        info = self._get_info()
        done = True
        return obs, reward, done, False, info
    else:
        reward = -1000  # Penalización por acción inválida
        done = True
        return self._get_obs(), reward, done, False, self._get_info()'''



    
    def step(self, action):
        # Verifica si la acción está permitida
        allowed_actions = allowed_function(self.current_root)
        if action in allowed_actions:
            # Aplica la mutación
            new_root = aplicate_mutation(self.current_root, action)
            self.current_root = prune_tree(self.current_root)
            new_erp_avg, _, _, _ = calculate_erp(new_root, clonar_arreglo_instancias(self.initial_instances))

            # Calcula la recompensa
            reward = self._evaluate_erp_change(self.prev_erp_avg, new_erp_avg)
            self.prev_erp_avg = new_erp_avg
            self.current_root = new_root.clone()

            # Seleccionar un nuevo nodo aleatorio 
            self.current_node = select_random_node(self.current_root)

            # Estado actual
            obs = self._get_obs()
            info = self._get_info()

            # Termina después de una acción
            done = True
            return obs, reward, done, False, info
        else:
            # Penalización por acción inválida
            reward =-5000# -5000
            done = True
            return self._get_obs(), reward, done, False, self._get_info()

    def render(self, mode='human'):
        #if self.estado_actual:
        #    print(self.estado_actual.get_root_node().to_string())
        if self.current_root:
            print(self.current_root.to_string())

##########################################################
###################### CLASE REWARD ########################
##########################################################

from stable_baselines3.common.env_checker import check_env

"""# Asume que TSPEnv es tu clase de entorno personalizada y que ya está definida
# Crear una instancia de tu entorno con los parámetros necesarios
def random_exploration(env, num_step):
    # Reset the environment to its initial state and get the initial observation (initial state)
    current_time_seed = time.time()
    observation, info = env.reset(seed=current_time_seed)
    print("Inicia episodio")
    print(env.current_root.to_string())
    print(env.prev_erp_avg)
    # Simulate the agent's actions for num_step time steps
    erps = []
    rewards = []
    for _ in range(num_step):
        # Choose a random action from the action space
        action = env.action_space.sample()
        print("Accion a realizar: ", action)
        # Take the chosen action and observe the resulting state, reward, and termination status
        observation, reward, terminated, truncated, info = env.step(action)

        # If the episode is terminated, reset the environment to the start cell
        if terminated:
            print("Fin episodio")
            print(env.current_root.to_string())
            erp_s_avg=env.prev_erp_avg
            print(observation , erp_s_avg)
            erps.append(erp_s_avg)
            rewards.append(reward)
            observation, info = env.reset()
    return rewards, erps"""

from stable_baselines3 import DQN
def evaluation_agent(env, model):
            # Reset the environment to its initial state and get the initial observation (initial state)
            observation, info = env.reset()
            # Simulate the agent's actions for num_step time steps
            reward_acum = 0
            while True:
                # Choose a random action from the action space
                action, _states = model.predict(observation, deterministic=True)
                print(action)
                # Take the chosen action and observe the resulting state, reward, and termination status
                observation, reward, terminated, truncated, info = env.step(action)
                reward_acum += reward

                if terminated:
                    erp_s_avg, _, erp_s_totals, _ = calcular_erp_arreglo_instancias(env.initial_instances)
                    return reward_acum, erp_s_avg,env.current_root.to_string(),erp_s_totals



                
initial_instances = leer_instancias2()  # Asegúrate de que esta función cargue las instancias iniciales adecuadamente
env = TSPEnv(initial_instances) 
'''
model = DQN(
    policy='MlpPolicy',
    env=env,
    learning_rate=0.001,  # Incremento
    exploration_fraction=0.5,  # Más rápido a la explotación
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,  # Más explotación
    batch_size=128,  # Tamaño mayor para estabilidad
    train_freq=1,  # Más frecuencia de entrenamiento
    gradient_steps=2,  # Más pasos de gradiente
    verbose=2
        )'''
model = DQN(
            policy='MlpPolicy',
            env=env,
            learning_rate=0.0011,
            exploration_fraction=0.9996,       # 10% de los pasos totales para disminuir epsilon
            exploration_initial_eps=1.0,    # Valor inicial de epsilon
            exploration_final_eps=0.0, 
            verbose=0
)


'''

start_time = time.time()
# Crear una instancia del callback
model.learn(total_timesteps=5000)
end_time = time.time()
execution_time = end_time - start_time
print('TIEMPO DE ENTRENAMIENTO: ', execution_time)
rewards, erps, algoritmo, erps_instancias = evaluation_agent(env, model)
print( f"ERP_AVG_EXEC: {erps} - ERP_INSTANCIA: {erps_instancias} - REWARD TOTAL: {rewards}") 
print( f"Algoritmo: {algoritmo}")
env.close()
'''
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import matplotlib.pyplot as plt
# Función para calcular promedio móvil
def moving_average(data, window_size):
    """
    Calcula el promedio móvil de los datos.
    :param data: Lista de valores originales.
    :param window_size: Tamaño de la ventana para el promedio móvil.
    :return: Lista con el promedio móvil aplicado.
    """
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean()

# Callback personalizado para recopilar métricas por paso
class MetricsCallback(BaseCallback):
    def __init__(self):
        super(MetricsCallback, self).__init__()
        self.erps_per_step = []  # ERP por paso
        self.rewards_per_step = []  # Recompensa por paso
        self.timesteps = []  # Timestep correspondiente

    def _on_step(self) -> bool:
        # Guardar ERP y recompensa del paso actual
        erp_step, _, _, _ = calcular_erp_arreglo_instancias(self.training_env.envs[0].initial_instances)
        reward_step = self.locals['rewards'][-1]  # Última recompensa (correspondiente al paso actual)
        
        # Guardar métricas
        self.erps_per_step.append(erp_step)
        self.rewards_per_step.append(reward_step)
        self.timesteps.append(self.num_timesteps)
        
        return True

# Crear el callback
metrics_callback = MetricsCallback()

# Entrenamiento del modelo
start_time = time.time()
model.learn(total_timesteps=100000, callback=metrics_callback)  # Incluye el callback
end_time = time.time()
execution_time = end_time - start_time
print('TIEMPO DE ENTRENAMIENTO: ', execution_time)

# Aplicar promedio móvil
window_size = 500
erps_moving_avg = moving_average(metrics_callback.erps_per_step, window_size)
rewards_moving_avg = moving_average(metrics_callback.rewards_per_step, window_size)

# Gráficos de las métricas con promedio móvil
plt.figure(figsize=(10, 6))
plt.plot(metrics_callback.timesteps, metrics_callback.erps_per_step, label="ERP por Paso", alpha=0.3, color="blue")
plt.plot(metrics_callback.timesteps, erps_moving_avg, label="Promedio Móvil ERP (500 pasos)", color="red", linewidth=2)
plt.title("Evolución del ERP por Paso con Promedio Móvil")
plt.xlabel("Timestep")
plt.ylabel("ERP")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(metrics_callback.timesteps, metrics_callback.rewards_per_step, label="Recompensa por Paso", alpha=0.3, color="orange")
plt.plot(metrics_callback.timesteps, rewards_moving_avg, label="Promedio Móvil Recompensa (500 pasos)", color="green", linewidth=2)
plt.title("Evolución de la Recompensa por Paso con Promedio Móvil")
plt.xlabel("Timestep")
plt.ylabel("Recompensa")
plt.legend()
plt.grid()
plt.show()
