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
    
    '''def three_opt(self):
        """
        Implementa el operador 3-opt para mejorar la solución actual.
        Se eliminan 3 aristas y se reconectan las rutas resultantes de la manera más óptima.
        """
        prev_diff = abs(self.optimal - self.current_distance)
        if len(self.cities_visited) < 6 or len(self.cities_to_visit) > 0:
            self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
            return False

        cities_visited_temp = list(self.cities_visited)
        best_cost = self.get_other_distance(cities_visited_temp)
        best_route = cities_visited_temp[:]

        for i in range(len(cities_visited_temp) - 5):
            for j in range(i + 2, len(cities_visited_temp) - 3):
                for k in range(j + 2, len(cities_visited_temp) - 1):
                    # Genera todas las combinaciones posibles para reconectar las 3 subrutas
                    new_routes = [
                        cities_visited_temp[:i + 1] +
                        cities_visited_temp[i + 1:j + 1][::-1] +
                        cities_visited_temp[j + 1:k + 1][::-1] +
                        cities_visited_temp[k + 1:]
                    ]
                    for new_route in new_routes:
                        new_cost = self.get_other_distance(new_route)
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_route = new_route[:]

        if best_cost < self.current_distance:
            self.cities_visited = best_route
            self.get_current_distance()
            self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
            return True
        else:
            self.modified = self.modified or (abs(self.optimal - self.current_distance) < prev_diff)
            return False'''

    

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
    def __init__(self, type_=None, value=None, index=None, parent=None):
        """
        Inicializa un nodo del árbol de sintaxis.
        
        :param type_: Tipo del nodo ('función' o 'método').
        :param value: Valor asociado al nodo (nombre de la función o método).
        :param index: Índice único del nodo.
        :param parent: Nodo padre (None si es el nodo raíz).
        """
        self.type = type_
        self.value = value
        self.index = index
        self.parent = parent
        self.left = None  # Nodo hijo izquierdo
        self.right = None  # Nodo hijo derecho
    def replace_function(self):
        """Reemplaza la función del nodo actual con otra función aleatoria."""
        if self.type == 'función':
            allowed_functions = ['if', 'and', 'or', 'while']
            new_function = random.choice([f for f in allowed_functions if f != self.value])
            #print(f"Función {self.value} reemplazada por: {new_function}")
            self.value = new_function

    def replace_terminal(self):
        """Reemplaza un nodo terminal por otro terminal aleatorio., 'opt2', 'opt2_2'"""
        if not self.left and not self.right and self.type == 'método':
            allowed_terminals = ['invert','opt2', 'swap','opt2_2', 'relocate']
            new_terminal = random.choice([t for t in allowed_terminals if t != self.value])
            #print(f"Nodo terminal reemplazado por: {new_terminal}")
            self.value = new_terminal
        else:
            if self.left:
                self.left.replace_terminal()
            if self.right:
                self.right.replace_terminal()
    
    
    def shrink_tree(self):
        """
        Reduce el árbol al nodo raíz, eliminando todos los subárboles
        y transformándolo en un nodo terminal.
        """
        # Verificar si el nodo actual es el nodo raíz (no tiene padre)
        if self.parent is None:
            # Convertir la raíz en un nodo terminal
            self.type = 'método'  # Cambiar el tipo a 'método'
            self.value = random.choice(['invert', 'opt2', 'opt2_2', 'swap', 'relocate'])  # Elegir un método aleatorio
            self.left = None  # Eliminar subárbol izquierdo
            self.right = None  # Eliminar subárbol derecho
        else:
            # Si no es el nodo raíz, intentar aplicar la operación al padre
            print("shrink_tree solo puede aplicarse al nodo raíz.")




    def add_random_subtree(self, max_depth=MAX_DEPTH):
        """Agrega un subárbol aleatorio al nodo actual si es terminal o un subárbol vacío."""
        if self.current_depth() >= max_depth - 1:
        # Si ya está en la profundidad máxima, no se puede agregar un subárbol
            return
        if not self.left and not self.right:
            allowed_functions = ['if', 'and', 'or', 'while']
            self.type = 'función'
            self.value = random.choice(allowed_functions)
            #print(f"Nodo terminal convertido en función: {self.value}")
            # Crear subárboles con profundidad aleatoria, controlada por max_depth
            self.left = create_syntax_tree(max_depth - 1, current_depth=1,index= self.index, parent=self)
            self.right = create_syntax_tree(max_depth - 1, current_depth=1,index= self.index, parent=self)
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
            #print("Subárboles intercambiados.")
        else:
            # Si alguno de los subárboles es None, intenta crear uno aleatorio para permitir el intercambio
            if not self.left:
                self.left = create_syntax_tree(2, current_depth=1,index= self.index, parent=self)
            if not self.right:
                self.right = create_syntax_tree(2, current_depth=1,index= self.index, parent=self)
            #print("Subárboles creados para el intercambio y luego intercambiados.")


    def mutate_subtree(self, depth, mutation_probability=0.5):
        """
        Aplica mutación al subárbol de este nodo hasta una profundidad específica.
        """
        if self.current_depth() >= MAX_DEPTH:
            return  # No mutar si ya estamos en la profundidad máxima
        # Validar que el nodo actual no sea inválido
        if self.type is None:
            return

        # Si la profundidad es 0 o el nodo no tiene hijos, realizar mutación directa
        if depth == 0 or (not self.left and not self.right):
            if self.type == 'función':
                self.replace_function()
            elif self.type == 'método':
                self.replace_terminal()
            return

        # Mutación aleatoria en los hijos o en el nodo actual
        if depth > 0 and random.random() < mutation_probability:
            if random.choice([True, False]) and self.left:
                self.left.mutate_subtree(depth - 1, mutation_probability)
            elif self.right:
                self.right.mutate_subtree(depth - 1, mutation_probability)
        else:
            # Si no se mutan los hijos, mutar el nodo actual
            if self.type == 'función':
                self.replace_function()
            elif self.type == 'método':
                self.replace_terminal()


    def execute(self, objetivo):
        """
        Ejecuta el nodo actual en el contexto del objetivo.
        Maneja nodos inválidos y garantiza seguridad durante la ejecución.
        """
        if self.type is None or self.value is None:
            # Si el nodo es inválido, retorna False para evitar errores.
            return False

        if self.type == 'método':
            try:
                # Ejecutar el método asociado al valor del nodo en el objetivo.
                return getattr(objetivo, self.value)()
            except AttributeError:
                # Si el método no existe en el objetivo, manejar el error.
                print(f"Error: El método '{self.value}' no existe en el objetivo.")
                return False
        elif self.type == 'función':
            # Evaluar el resultado del hijo izquierdo.
            left_result = self.left.execute(objetivo) if self.left else False

            if self.value == "if":
                # Si la condición izquierda es verdadera, ejecutar el hijo derecho.
                return self.right.execute(objetivo) if left_result and self.right else False
            elif self.value == "and":
                # AND: Si el izquierdo es falso, retornar falso directamente.
                if not left_result:
                    return False
                return self.right.execute(objetivo) if self.right else False
            elif self.value == "or":
                # OR: Si el izquierdo es verdadero, retornar verdadero directamente.
                if left_result:
                    return True
                return self.right.execute(objetivo) if self.right else False
            elif self.value == "while":
                # Límite de iteraciones para evitar bucles infinitos.
                iterations_limit = len(objetivo.cities_visited)

                for _ in range(iterations_limit):
                    # Evaluar el nodo izquierdo (condición del while).
                    left_result = self.left.execute(objetivo) if self.left else False

                    # Si la condición es falsa, salir del bucle.
                    if not left_result:
                        return False

                    # Evaluar el nodo derecho (cuerpo del while).
                    right_result = self.right.execute(objetivo) if self.right else False

                    # Si el cuerpo devuelve True, retornar True.
                    if right_result:
                        return True

                # Si se alcanzó el límite de iteraciones sin retornar True, retornar False.
                return False

        # Si el nodo no es ni método ni función reconocida, retornar False.
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
        #if self.type is None or self.value is None:
            # Si el nodo no es válido, retorna un nodo por defecto
            #return Node(type_='método', value=random.choice(['invert', 'opt2', 'opt2_2', 'swap', 'relocate']), index=index, parent=parent)
        if self.type is None:
           return None

        executable_node = Node(self.type, self.value, self.index)

        if self.left:
            executable_node.left = self.left.get_executable_tree()
            if executable_node.left:
                executable_node.left.parent = executable_node
        else:
            if self.type == 'función':
                executable_node.left = Node(type_='método', value='invert', parent=executable_node)

        if self.right:
            executable_node.right = self.right.get_executable_tree()
            if executable_node.right:
                executable_node.right.parent = executable_node
        else:
            if self.type == 'función':
                executable_node.right = Node(type_='método', value='invert', parent=executable_node)
        # Validar integridad del árbol ejecutable
        validate_tree_completeness(executable_node)

        return executable_node




    
    def clone(self):
        """Clona el nodo y todos sus descendientes."""
        # Crear un nuevo nodo vacío
        cloned_node = Node()
        
        # Copiar los atributos del nodo actual
        cloned_node.type = self.type
        cloned_node.value = self.value
        cloned_node.index = self.index
        cloned_node.parent = None  # Evitar referencias circulares iniciales
        
        # Clonar los hijos recursivamente si existen
        if self.left and self.left != self:
            cloned_node.left = self.left.clone()
            cloned_node.left.parent = cloned_node
        else:
            cloned_node.left = None

        if self.right and self.right != self:
            cloned_node.right = self.right.clone()
            cloned_node.right.parent = cloned_node
        else:
            cloned_node.right = None

        return cloned_node

    def clone2(self):
        """Clona el nodo y todos sus descendientes."""
        cloned_node = Node(self.type, self.value, self.index)
        cloned_node.parent = None  # Evitar referencias circulares
        
        # Clonar hijos recursivamente
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

def validate_tree_completeness(root_node):
    """
    Valida que todos los nodos de tipo 'función' tengan hijos válidos.
    Si no los tienen, los rellena con nodos predeterminados.
    """
    if not root_node:
        return

    if root_node.type == 'función':
        if not root_node.left:
            root_node.left = Node(type_='método', value='invert')
            root_node.left.parent = root_node
        if not root_node.right:
            root_node.right = Node(type_='método', value='invert')
            root_node.right.parent = root_node

    if root_node.left:
        validate_tree_completeness(root_node.left)
    if root_node.right:
        validate_tree_completeness(root_node.right)



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

def create_syntax_tree(max_depth, current_depth, index, parent):
    """
    Crea un árbol de sintaxis de forma recursiva, asegurando que todos los nodos sean válidos
    y que la profundidad máxima no se sobrepase.
    """
    # Si se alcanza la profundidad máxima, crear un nodo terminal.

    if current_depth >= max_depth:
        #return Node(type_='método', value=random.choice(['invert', 'opt2', 'opt2_2', 'swap', 'relocate']), index=index, parent=parent)
        return None

    # Decidir si será terminal o función.
    if current_depth < max_depth - 1:
        node_type = 'función'
        node_value = random.choice(['if', 'and', 'or', 'while'])
    else:
        # Último nivel válido para funciones, solo métodos en el próximo nivel.
        node_type = 'método'
        node_value = random.choice(['invert', 'opt2', 'opt2_2', 'swap', 'relocate'])

    # Crear el nodo actual.
    node = Node(type_=node_type, value=node_value, index=index, parent=parent)

    # Si es una función, asegurar que tenga hijos válidos.
    if node_type == 'función':
        node.left = create_syntax_tree(max_depth, current_depth + 1, index=2 * index + 1, parent=node)
        node.right = create_syntax_tree(max_depth, current_depth + 1, index=2 * index + 2, parent=node)

    return node



def prune_tree(node, max_depth=MAX_DEPTH):
    """
    Podar el árbol asegurando que no exceda la profundidad máxima.
    """
    if not node or node.current_depth() >= max_depth:
        return None
     # Si el nodo es una función pero le falta uno o ambos hijos
    if node.type == 'función' and (not node.left or not node.right):
        # Rellenar con nodos terminales predeterminados
        if not node.left:
            node.left = Node(type_='método', value=random.choice(['invert', 'opt2', 'opt2_2', 'swap', 'relocate']))
            node.left.parent = node
        if not node.right:
            node.right = Node(type_='método', value=random.choice(['invert', 'opt2', 'opt2_2', 'swap', 'relocate']))
            node.right.parent = node

    # Recursivamente podar los subárboles izquierdo y derecho
    node.left = prune_tree(node.left, max_depth)
    node.right = prune_tree(node.right, max_depth)

   
    return node

    

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
            #print("Aplicando mutación: replace_terminal")
            root_node.replace_terminal()
       # else:
            #print("Mutación replace_terminal no es válida en el estado actual del árbol.")
            
    elif mutation_index == 1:
        # Mutación: shrink_tree
        if root_node.depth() > 1:
            #print("Aplicando mutación: shrink_tree")
            root_node.shrink_tree()
        #else:
            #print("Mutación shrink_tree no es válida porque la profundidad del árbol es 1 o menor.")
            
    elif mutation_index == 2:
        # Mutación: add_random_subtree
        if root_node.is_full():
            #print("Aplicando mutación: add_random_subtree")
            root_node.add_random_subtree(max_depth=MAX_DEPTH)
       # else:
            #print("Mutación add_random_subtree no es válida, ya que el nodo actual no permite añadir un subárbol.")
            
    elif mutation_index == 3:
        # Mutación: swap_subtrees
        if root_node.left and root_node.right:
            #print("Aplicando mutación: swap_subtrees")
            root_node.swap_subtrees()
        #else:
            #print("Mutación swap_subtrees no es válida, ya que no hay subárboles para intercambiar.")
            
    elif mutation_index == 4:
        # Mutación: mutate_subtree
        if root_node.depth() > 1 and root_node.depth()< MAX_DEPTH:
            #print("Aplicando mutación: mutate_subtree")
            root_node.mutate_subtree(depth=2, mutation_probability=0.5)
       # else:
            #print("Mutación mutate_subtree no es válida porque la profundidad del árbol es insuficiente.")

    elif mutation_index == 5:
        # Mutación: replace_function
        if root_node.is_executable():
            #print("Aplicando mutación: replace_function")
            root_node.replace_function()
        #else:
            #print("Mutación replace_function no es válida en el estado actual del árbol.")
    
    else:
        print(f"Índice de mutación no válido: {mutation_index}")
     # Validar la integridad del árbol después de la mutación
    validate_tree_completeness(root_node)
    
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
            print(f"Instance actual: {instance}")
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
    print(f"ERP Calculado: {erp_s_avg}, Totales por Instancia: {erp_s_totals}")
    return erp_s_avg, objetivos_temp, erp_s_totals, distances

def execute_root(root,objetivos):
    obj = clonar_arreglo_instancias(objetivos)
    root_executable = root.get_executable_tree()
    for objetivo in obj:
        root_executable.execute(objetivo)
    return obj

'''def crear_arreglo_instancias(files):
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
  return instancias'''
def crear_arreglo_instancias(files):
    instancias = []
    for f in files:
        try:
            # Abrir el archivo y leer las líneas
            with open(f, 'r') as file:
                lines = file.readlines()
            
            # Mostrar contenido del archivo para depuración
            print(f"\nContenido del archivo {f}:")
            for idx, line in enumerate(lines):
                print(f"Línea {idx}: {line.strip()}")
            
            # Procesar las líneas
            if len(lines) < 3:
                print(f"Advertencia: El archivo {f} no tiene suficientes líneas.")
                continue  # Saltar archivos con contenido insuficiente

            optimo = float(lines[0].strip())  # Primera línea: valor óptimo
            n = int(lines[1].strip())  # Segunda línea: número de ciudades
            
            # Tercera línea en adelante: matriz de distancias
            matrix = []
            for line in lines[2:]:
                if line.strip():  # Ignorar líneas vacías
                    matrix.append(list(map(float, line.split())))

            # Validar tamaño de la matriz
            if len(matrix) != n or any(len(row) != n for row in matrix):
                print(f"Advertencia: Matriz malformada en el archivo {f}. Se esperaba {n}x{n}, pero se obtuvo {len(matrix)}x{len(matrix[0])}.")
                continue  # Saltar archivos con matrices malformadas

            # Crear instancia
            instance = Instance(f, optimo, n, matrix)
            instance.generate_initial_solution_2()
            instance.get_current_distance()
            instancias.append(instance)

        except ValueError as e:
            print(f"Error procesando el archivo {f}: {e}")
        except Exception as e:
            print(f"Error inesperado en el archivo {f}: {e}")
    
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
    instance.generate_initial_solution_2()
    instance.get_current_distance()
    instancias.append(instance)
  return instancias

def leer_instancias2():
    #fileTrain = ["./prueba_instancias/att48.txt","./prueba_instancias/berlin52.txt","./prueba_instancias/eil51.txt","./prueba_instancias/pr76.txt","./prueba_instancias/st70.txt"]
    '''fileTrain = [    "./Matrices/a280.txt",
    "./Matrices/ch130.txt",
    "./Matrices/ch150.txt",
    "./Matrices/eil101.txt",
    "./Matrices/eil51.txt",
    "./Matrices/eil76.txt",
    "./Matrices/gil262.txt",
    "./Matrices/gr48.txt",
    "./Matrices/pa561.txt",
    "./Matrices/rat575.txt",
    "./Matrices/rd100.txt",
    "./Matrices/rd400.txt",
    "./Matrices/st70.txt"]'''
    fileTrain = ["./Matrices/pr152.txt","./Matrices/vm1084.txt"]
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

def get_tree_state_representation(root_node):
    if not root_node:
        return [0] * 13  # Vector de tamaño fijo con ceros si el árbol está vacío

    # Inicializar contadores y variables
    max_depth = 0
    function_counts = defaultdict(int)
    method_counts = defaultdict(int)
    total_nodes = 0
    current_node_index = -1  # Valor predeterminado
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

        # Validar que el índice no sea None antes de comparar
        if current_node.index is not None and current_node.index > current_node_index:
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
        method_counts['opt2'],
        method_counts['opt2_2'],
        method_counts['swap'],                   # Conteo de métodos 'swap'
        method_counts['relocate'],               # Conteo de métodos 'relocate'
        current_node_index if current_node_index != -1 else 0,  # Índice del nodo actual
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
        if root_node.depth() > 1 and root_node.depth()<MAX_DEPTH:
            allowed_actions.append(1)  # Acción: shrink_tree
        
        # Verifica si se puede aplicar add_random_subtree (requiere que el nodo esté completo)
        if root_node.is_full() and root_node.depth()<MAX_DEPTH:
            allowed_actions.append(2)  # Acción: add_random_subtree
        
        # Verifica si se puede aplicar swap_subtrees (requiere que el nodo tenga ambos subárboles)
        if root_node.left and root_node.right  and root_node.depth()<MAX_DEPTH:
            allowed_actions.append(3)  # Acción: swap_subtrees
        
        # Verifica si se puede aplicar mutate_subtree (requiere una profundidad mayor que 1)
        if root_node.depth() > 1  and root_node.depth()<MAX_DEPTH:
            allowed_actions.append(4)  # Acción: mutate_subtree
        
        # Validación adicional para una acción de reemplazo de función (acción 5, si es aplicable)
        if root_node.type == 'función'  and root_node.depth()<MAX_DEPTH:
            allowed_actions.append(5)  # Acción: replace_function
        # Si no hay acciones válidas, devuelve una acción predeterminada
        if not allowed_actions:
            allowed_actions.append(-1)  # Acción especial para evitar penalizaciones excesivas

        return allowed_actions


def calculate_reward(delta, erp_anterior, num_nodes, current_erp, repeated_actions=None, action=None):
    reward = 0

    # Bonificación o penalización basada en el cambio de ERP
    if delta > 0:
        reward += 100 * delta  # Recompensa por mejora proporcional al progreso
    else:
        reward -= 75 * abs(delta)  # Penalización por empeorar, más leve que antes

    # Bonificaciones escalonadas por alcanzar ERPs bajos
    if current_erp < 0.01:
        reward += 500  # Bonificación máxima por ERPs extremadamente bajos
    elif current_erp < 0.02:
        reward += 400
    elif current_erp < 0.03:
        reward += 300
    elif current_erp < 0.05:
        reward += 200
    elif current_erp < 0.1:
        reward += 50

    # Penalización por falta de progreso (evita estancamiento)
    if abs(delta) < 1e-3:
        reward -= 25  # Penalización leve

    # Penalización por tamaño del árbol (fomenta simplicidad)
    reward -= num_nodes * 0.01  # Penalización proporcional al tamaño, reducida

    # Penalización por acciones repetidas sin mejora significativa
    if repeated_actions and action in repeated_actions and delta < 1e-3:
        reward -= 50  # Penalización ajustada

    # Bonificación por alcanzar el objetivo final
    if current_erp < 0.03 and delta > 0:
        reward += 100

    # Normalización del rango de recompensas
    return max(-1000, min(1000, reward))  # Rango ajustado a [-1000, 1000]


##########################################################
###################### CLASE ENV ########################
##########################################################
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete

class TSPEnv(gym.Env):
    """Entorno personalizado para el problema TSP."""
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_instances, seed=None, no_progress_limit=5, erp_threshold=0.03):
        super(TSPEnv, self).__init__()
        # Instancias iniciales del TSP
        self.repeated_actions = deque(maxlen=6)  # Historial de las últimas 6 acciones
        self.initial_instances = initial_instances
        self.current_root = create_syntax_tree(max_depth=MAX_DEPTH-4, current_depth=0, index=0, parent=None)  # Crear el árbol una sola vez
        self.estado_actual = None
        self.previous_root = self.current_root.clone() 
        self.current_node = None  # Nodo actual seleccionado
        self.prev_erp_avg, _, _, _ = calculate_erp(
            self.current_root,
            clonar_arreglo_instancias(self.initial_instances)
        )

        # Espacio de acciones
        self.action_space = Discrete(6)  # Acciones
        self.observation_space = Box(low=0, high=100, shape=(13,), dtype=np.float32)

        # Parámetros de entrenamiento
        self.max_steps_per_episode = 20
        self.steps_taken = 0
        self.steps_without_progress = 0  # Contador de pasos sin mejora
        self.no_progress_limit = no_progress_limit  # Límite de pasos sin mejora
        self.erp_threshold = erp_threshold  # Umbral de ERP para terminar episodios

        # Historial de estados
        self.state_history = deque(maxlen=2)
        empty_state = np.zeros((13,), dtype=np.int32)
        self.state_history.extend([empty_state, empty_state])

        # Mapeo de acciones
        self._actions = {
            0: 'replace_terminal',
            1: 'shrink_tree',
            2: 'add_random_subtree',
            3: 'swap_subtrees',
            4: 'mutate_subtree',
            5: 'replace_function' 
        }

        # Inversa del mapeo de acciones
        self._actions2 = {v: k for k, v in self._actions.items()}
    

    def reset(self, seed=None):
        # Si el mejor ERP es suficientemente bajo, reutiliza el árbol anterior
        if self.prev_erp_avg < 0.05:  # Si el ERP final es bajo, usa el mismo árbol
            self.current_root = self.previous_root.clone()
        else:
            print("Generando un nuevo árbol.")
            self.current_root = create_syntax_tree(max_depth=MAX_DEPTH-4, current_depth=0, index=0, parent=None)

        validate_tree_completeness(self.current_root)

        # Calcular el ERP inicial
        self.prev_erp_avg, _, _, _ = calculate_erp(
            self.current_root,
            clonar_arreglo_instancias(self.initial_instances)
        )
        print(f"ERP Inicial: {self.prev_erp_avg:.6f}")

        # Reiniciar contadores
        self.steps_taken = 0
        self.steps_without_progress = 0

        return self._get_obs(), self._get_info()




    def _get_obs(self):
        # Obtén el estado actual del árbol
        current_state = get_tree_state_representation(self.current_root)
        return np.array(current_state, dtype=np.float32)
    
    def _get_info(self):
        # Información adicional del entorno
        return {
            'tree_structure': self.current_root.to_string(),
            'erp': self.prev_erp_avg,
            'state': self._get_obs()
        }

    def step(self, action):
        # Verificar si la acción está permitida
        allowed_actions = allowed_function(self.current_root)
        if action in allowed_actions:
            # Aplicar la mutación al árbol actual
            new_root = aplicate_mutation(self.current_root, action)
            self.current_root = prune_tree(new_root)

            validate_tree_completeness(self.current_root)

            # Calcular el nuevo ERP
            new_erp_avg, _, _, _ = calculate_erp(
                self.current_root,
                clonar_arreglo_instancias(self.initial_instances)
            )

            # Calcular el delta entre el ERP anterior y el nuevo
            delta = self.prev_erp_avg - new_erp_avg

            # **Evitar que el ERP empeore**
            if delta < 0:  # Si el ERP empeora
                print(f"ERP empeoró. Manteniendo el árbol anterior.")
                self.current_root = self.previous_root.clone()  # Volver al árbol anterior
                reward = -50  # Penalización por empeorar
            else:  # Si el ERP mejora o se mantiene
                reward = calculate_reward(delta, self.prev_erp_avg, len(get_tree_state_representation(self.current_root)),
                                        new_erp_avg, self.repeated_actions, action)
                self.previous_root = self.current_root.clone()  # Guardar el mejor árbol hasta ahora

            # Actualizar el ERP previo para el próximo paso
            self.prev_erp_avg = new_erp_avg

        else:
            # Penalización por acción no válida
            reward = -100
            self.steps_without_progress += 1
        # Actualizar el historial de acciones
        self.repeated_actions.append(action)
        if len(self.repeated_actions) > 50:  # Limitar el historial a las últimas 50 acciones
            self.repeated_actions.pop(0)
        # Incrementar el contador de pasos
        self.steps_taken += 1

        # Determinar si se ha alcanzado el número máximo de pasos o no hay progreso
        done = (
            self.steps_taken >= self.max_steps_per_episode or
            self.steps_without_progress >= self.no_progress_limit or
            self.prev_erp_avg <= self.erp_threshold
        )
         # Finalizar el episodio si se alcanza el ERP objetivo
        if self.prev_erp_avg <= 0.05:
            done = True

        if done:
        # Guarda el árbol final del episodio actual para usarlo en el próximo episodio
            validate_tree_completeness(self.current_root)
            self.previous_root = self.current_root.clone()

        return self._get_obs(), reward, done, False, self._get_info()





    def render(self, mode='human'):
        # Renderizar la estructura del árbol
        if self.current_root:
            print(self.current_root.to_string())
    
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
    
    
    def _evaluate_erp_change(self, erp_anterior, erp_posterior):
        delta = erp_anterior - erp_posterior  # Cambio en el ERP
        reward = calculate_reward(delta, erp_anterior)

        # Penalización adicional por tamaño del árbol (opcional)
        num_nodes = len(get_tree_state_representation(self.current_root))
        size_penalty = num_nodes * 0.002
        reward -= size_penalty
        return reward


from stable_baselines3.common.env_checker import check_env
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



                
#initial_instances = leer_instancias2()  # Asegúrate de que esta función cargue las instancias iniciales adecuadamente
#env = TSPEnv(initial_instances) 

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

def evaluate_final_erp(env, model, current_time_seed):
    """
    Evalúa el ERP final utilizando el algoritmo aprendido por el modelo.
    
    :param env: El entorno TSP.
    :param model: El modelo entrenado.
    :param current_time_seed: Semilla para reiniciar el entorno.
    :return: Una tupla que contiene:
        - ERP promedio final (float)
        - ERP por instancia (list)
        - Representación del árbol aprendido (string)
    """
    # Reiniciar el entorno con la semilla proporcionada
    observation, info = env.reset(seed=current_time_seed)
    
    # Ejecutar el algoritmo aprendido sobre las instancias
    while True:
        # Elegir una acción basada en el modelo entrenado
        action, _states = model.predict(observation, deterministic=True)
        
        # Tomar la acción en el entorno
        observation, reward, terminated, truncated, info = env.step(action)

        # Detener si el episodio termina
        if terminated:
            break

    # Evaluar el ERP final utilizando el árbol generado
    final_erp_avg, _, final_erp_totals, _ = calcular_erp_arreglo_instancias(env.initial_instances)
    
    # Obtener la representación del árbol final
    final_algorithm = env.current_root.to_string()

    return final_erp_avg, final_erp_totals, final_algorithm

def evaluate_final_tree(env, tree):
    """
    Evalúa un árbol final en las instancias iniciales del entorno.
    
    :param env: Entorno TSP personalizado.
    :param tree: Árbol final generado durante el entrenamiento.
    :return: ERP promedio final y ERP por instancia.
    """
    # Clonar las instancias para evitar modificar las originales
    test_instances = clonar_arreglo_instancias(env.initial_instances)
    
    # Ejecutar el árbol en las instancias
    #tree_executable = tree.get_executable_tree()  # Asegúrate de que el árbol sea ejecutable
    for instance in test_instances:
        tree.execute(instance)
    
    # Calcular ERP promedio y por instancia
    final_erp_avg, _, final_erp_totals, _ = calcular_erp_arreglo_instancias(test_instances)
    return final_erp_avg, final_erp_totals


import os
import csv
from stable_baselines3.common.callbacks import BaseCallback
class MetricsCallback(BaseCallback):
    def __init__(self, save_dir='.', verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.save_dir = save_dir
        self.episode_rewards = []
        self.epsilon_values = []
        self.episode_erps = []  # Lista para almacenar los ERPs
        self.episode_algorithms = []  # Lista para almacenar los algoritmos
        self.episode_reward = 0.0
        self.episode_steps = 0
        csv_path = os.path.join(self.save_dir, 'training_metrics.csv')
        self.csv_file = open(csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file, delimiter=";")
        self.csv_writer.writerow(['Episode', 'Reward', 'ERP', 'Epsilon', 'Algorithm'])  # Añadimos 'Algorithm' al CSV

    def _on_step(self) -> bool:
        # Obtener la recompensa actual y el indicador done
        reward = self.locals['rewards']
        done = self.locals['dones']
        epsilon = self.model.exploration_rate
        self.epsilon_values.append(self.model.exploration_rate)
        # Acumular la recompensa
        self.episode_reward += reward[0]
        self.episode_steps += 1
        if done[0]:
            # Acceder a la información del entorno
            erp_s_avg = self.locals['infos'][0]['erp']
            alg = self.locals['infos'][0]['tree_structure']
            # Registrar las métricas
            self.episode_rewards.append(self.episode_reward)
            self.epsilon_values.append(epsilon)
            self.episode_erps.append(erp_s_avg)
            self.episode_algorithms.append(alg)
            # Escribir en el archivo CSV
            self.csv_writer.writerow([len(self.episode_rewards), self.episode_reward, erp_s_avg, epsilon, alg])
            if self.verbose > 0:
                print(f"***Episodio {len(self.episode_rewards)}: Recompensa = {self.episode_reward}, ERP = {erp_s_avg}, Epsilon = {epsilon}***")
            # Reiniciar las variables
            self.episode_reward = 0.0
            self.episode_steps = 0
        return True

    def _on_training_end(self):
        # Cerrar el archivo CSV al finalizar el entrenamiento
        self.csv_file.close()


from sb3_contrib import QRDQN
import torch
if __name__ == '__main__':
    start_time = time.time()  # Inicio del tiempo total
    TIME_STEPS =1000  # Número de pasos de entrenamiento por experimento
    ITERATIONS = 1  # Número de experimentos
    window_size = 500  # Tamaño de la ventana para el promedio móvil
    experiment_results = []  # Lista para almacenar resultados de todos los experimentos
        # Definir la arquitectura personalizada de la política
    policy_kwargs = dict(
        net_arch=[128, 128],  # Capas ocultas
        activation_fn=torch.nn.ReLU  # Activación ReLU
    )


    for iteration in range(ITERATIONS):
        print(f"\n===== Iniciando Experimento {iteration + 1} =====")

       
        # Crear una carpeta para guardar los modelos
        model_save_dir = "./saved_models"
        os.makedirs(model_save_dir, exist_ok=True)

        # Generar una semilla única para el experimento actual
        current_seed = random.randint(0, 1_000_000)

        # Inicializar el entorno con la semilla
        initial_instances = leer_instancias2()
        env = TSPEnv(initial_instances)
        observation, info = env.reset(seed=current_seed)

        # Crear el modelo DQN
        model = DQN(
            policy='MlpPolicy',
            env=env,
            learning_rate=0.0011,# 0.00011
            #policy_kwargs=policy_kwargs, 
            exploration_fraction=0.9996,       # Exploración decreciente0.9996
            exploration_initial_eps=1.0,       # Epsilon inicial
            exploration_final_eps=0.0,         # Epsilon final0.0 0
            verbose=0
        )
        
        # Crear el callback
        metrics_callback = MetricsCallback()

        # Entrenar el modelo
        start_training_time = time.time()  # Inicio del tiempo de entrenamiento
        model.learn(total_timesteps=TIME_STEPS, callback=metrics_callback)  # Incluye el callback
        end_training_time = time.time()  # Fin del tiempo de entrenamiento
        
        # Guardar el modelo y la semilla
        model_save_path = os.path.join(model_save_dir, f"TSP_model_seed_{current_seed}.zip")
        model.save(model_save_path)
        print(f"Modelo guardado en {model_save_path} con semilla {current_seed}")

        
        # Registrar la semilla en un archivo CSV
        seed_log_path = os.path.join(model_save_dir, "seed_log.csv")
        with open(seed_log_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([f"TSP_model_seed_{current_seed}.zip", current_seed])
        print(f"Semilla registrada: {current_seed}")

        # Guardar el árbol final generado
        final_tree = env.current_root
        print(f"Árbol final aprendido:\n{final_tree.to_string()}")

        # Evaluar el ERP final con el árbol aprendido
        final_erp_avg_tree, final_erp_totals_tree = evaluate_final_tree(env, final_tree)
        print(f"\nEvaluación del árbol final:")
        print(f"ERP Promedio Final (Árbol): {final_erp_avg_tree}")
        print(f"ERP por Instancia (Árbol): {final_erp_totals_tree}")

        # Evaluar el ERP final utilizando el modelo aprendido
        current_time_seed = time.time()  # Semilla para la evaluación
        final_erp_avg_model, final_erp_totals_model, final_algorithm = evaluate_final_erp(env, model, current_time_seed)
        print(f"\nEvaluación Final con Modelo:")
        print(f"ERP por Instancia (Modelo): {final_erp_totals_model}")
        print(f"Algoritmo Final Utilizado: {final_algorithm}")


        end_time = time.time()  # Fin del tiempo total
        execution_time = end_time - start_time  # Cálculo del tiempo total
        print(f"TIEMPO TOTAL DE EJECUCIÓN: {execution_time:.2f} segundos")

        # Guardar los resultados de este experimento
        experiment_results.append({
            'iteration': iteration + 1,
            'execution_time': execution_time,
            'erp_avg_tree': final_erp_avg_tree,
            'erp_totals_tree': final_erp_totals_tree,
            'erp_avg_model': final_erp_avg_model,
            'erp_totals_model': final_erp_totals_model,
            'algorithm': final_algorithm
        })
        
        # Calcular y graficar el promedio móvil
        erps_series = pd.Series(metrics_callback.episode_erps)
        rewards_series = pd.Series(metrics_callback.episode_rewards)
        erps_moving_avg = erps_series.rolling(window=window_size, min_periods=1).mean()
        rewards_moving_avg = rewards_series.rolling(window=window_size, min_periods=1).mean()

        # Graficar ERP por Episodio
        plt.figure(figsize=(10, 6))
        plt.plot(erps_series.index + 1, erps_series, label="ERP por Episodio", alpha=0.3, color="blue")
        plt.plot(erps_series.index + 1, erps_moving_avg, label=f"Promedio Móvil ERP ({window_size} episodios)", color="red", linewidth=2)
        plt.title(f"Evolución del ERP por Episodio (Experimento {iteration + 1})")
        plt.xlabel("Episodio")
        plt.ylabel("ERP")
        plt.legend()
        plt.grid()
        plt.show()

        # Graficar Recompensa por Episodio
        plt.figure(figsize=(10, 6))
        plt.plot(rewards_series.index + 1, rewards_series, label="Recompensa por Episodio", alpha=0.3, color="orange")
        plt.plot(rewards_series.index + 1, rewards_moving_avg, label=f"Promedio Móvil Recompensa ({window_size} episodios)", color="green", linewidth=2)
        plt.title(f"Evolución de la Recompensa por Episodio (Experimento {iteration + 1})")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.legend()
        plt.grid()
        plt.show()

        # Graficar Decaimiento de Epsilon
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(metrics_callback.epsilon_values) + 1), metrics_callback.epsilon_values, label=f"Epsilon (Iteración {iteration + 1})", color="blue")
        plt.title(f"Decaimiento de Epsilon por Episodio (Experimento {iteration + 1})")
        plt.xlabel("Episodio")
        plt.ylabel("Epsilon")
        plt.legend()
        plt.grid()
        plt.show()


        # Liberar recursos del modelo y entorno
        del model
        env.close()

    # Almacenar el ERP final del modelo al terminar el entrenamiento
    last_erp_value = metrics_callback.episode_erps[-1]  # Último ERP registrado por el callback

    # Imprimir el ERP final del árbol y del modelo
    print("\n=== Evaluación Final con Modelo ===")
    print(f"ERP Final (Modelo): {last_erp_value:.6f}")  # ERP actualizado más reciente
    print(f"Algoritmo Final Utilizado: {env.current_root.to_string()}")

    print("\n===== Resumen Final de Resultados =====")
    print(f"Iteración {iteration + 1} - ERP Árbol: {final_erp_avg_tree:.6f}, ERP Final (Modelo): {last_erp_value:.6f}")

# Evaluar el árbol final con todas las instancias
final_erp_avg, _, final_erp_totals, distances = calculate_erp(env.current_root, env.initial_instances)

# Imprimir los resultados
print("\nEvaluación Final del Árbol con Algoritmo Final:")
print(f"ERP Promedio Final: {final_erp_avg:.6f}")
print(f"ERPs por Instancia: {final_erp_totals}")
print(f"Distancias por Instancia: {distances}")

