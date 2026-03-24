import numpy as np
try:
    from . import tree_core
except ImportError:
    import tree_core

def calculate_gini(y):
    '''
    Calcula la impureza de Gini para un vector de etiquetas y usando el motor hecho en C++
    '''
    if len(y) == 0: # Si no hay muestras en el nodo actual
        return 0.0 # Devuelve 0
    
    # En otro caso, convierto los tipos para poder pasarlo a C++
    y_cpp = np.ascontiguousarray(y).astype(np.int32)
    
    # Llamamos a la función de C++
    return tree_core.calcular_gini(y_cpp)

class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        '''
        Feature refiere al índice de la columna usada para separar
        Threshold refiere al valor de corte de la feature
        Left refiere al nodo hijo de la izquierda
        Right refiere al nodo hijo de la derecha
        Value refiere a la clase predicha
        '''
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:

    def __init__(self, min_samples=2, max_depth=200):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        # Acá hacemos la construcción recursiva
        self.root = self._grow_tree(X, y)

    def _information_gain(self, y, y_izq, y_der, parent_gini):
        '''
        La función aplica la media ponderada del gini.
        '''
        # Obtengo pesos (proporción de la división)
        n = len(y)
        n_izq = len(y_izq)
        n_der = len(y_der)

        # Obtengo los gini actuales
        gini_izq = calculate_gini(y_izq)
        gini_der = calculate_gini(y_der)

        # Hago la ponderación del gini medio
        gini_medio = (n_izq / n) * gini_izq + (n_der / n) * gini_der

        # Devolvemos la mejora respecto del parent gini
        return parent_gini - gini_medio

    def _best_split(self, X, y):
        '''
        La función devuelve el índice de la mejor columna,
        el mejor umbral y la ganancia obtenida.
        '''
        best_gain = -1
        split_idx = -1
        split_threshold = -1

        parent_gini = calculate_gini(y)

        for columna in range(X.shape[1]): # Por cada variable clínica
            valores_columna = X[:, columna] # Me quedo con todos los valores

            umbrales_unicos = np.unique(valores_columna) # Busco los valores únicos (para probarlos como umbrales)

            for threshold in umbrales_unicos: # Por cada uno de estos valores únicos
                mask = valores_columna <= threshold # Creo una máscara divisoria (los valores por encima y los de debajo)

                y_izq = y[mask] # Los que están por debajo o igual al threshold
                y_der = y[~mask] # Los que pasan el threshold

                # Un condicional para el caso de que el corte no haya dividido a nadie
                if len(y_izq) == 0 or len(y_der) == 0:
                    continue

                gain = self._information_gain(y, y_izq, y_der, parent_gini)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = columna
                    split_threshold = threshold
        
        return split_idx, split_threshold

    def _most_common_label(self, y):
        '''
        Función utilizada para _grow_tree para calcular la clase más fecuente en un nodo.
        '''
        return np.bincount(y).argmax() # argmax devuelve el índice con el valor más alto (la moda)

    def _grow_tree(self, X, y, depth=0):
        '''
        Función recursiva que se llama a si misma para ir creando las ramas del árbol
        '''
        n_samples, n_features = X.shape # Me quedo con el número de filas y el número de columnas
        n_labels = len(np.unique(y)) # Son la cantidad de clases a las que puede pertenercer una muestra
        
        # Condicional que cubre tres casos: Hay solo una clase, llegamos al tope de profundidad del árbol, o quedan pocos pacientes
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value) # Hago que esta rama devuelva la clase más común dentro de esa división
        
        # Búsqueda del mejor corte
        feat_idx, threshold = self._best_split(X, y)

        # Si no se encuentra una división mejor que la actual
        if feat_idx == -1:
            return Node(value=self._most_common_label(y))

        # Creación de las divisiones o nodos hijos
        '''
        Para hacerlo agarro la todas las filas de la columna que se encontró como mejor corte
        uso el threshold que se encontró para el mejor corte
        Hago una máscara booleana para los valores menores que el threshold (los del nodo de la izquierda)
        La misma máscara negada representa a las muestras que se fueron al nodo de la derecha.
        '''
        idx_izq = X[:, feat_idx] <= threshold
        idx_der = ~idx_izq

        # Limpiaré de las máscaras los valores falsos
        nodo_izq = X[idx_izq, :]
        y_izq = y[idx_izq]

        nodo_der = X[idx_der, :]
        y_der = y[idx_der]
        
        # Llamamos a la función recursiva para la rama izquierda y para la derecha
        left_child = self._grow_tree(nodo_izq, y_izq, depth + 1)
        right_child = self._grow_tree(nodo_der, y_der, depth + 1)

        return Node(feature=feat_idx, threshold=threshold, left=left_child, right=right_child)


    def predict(self, X):
        '''
        Recibe una matriz X de pacientes y devuelve un array con las predicciones
        '''
        # Pasamos a cada paciente por el árbol
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        '''
        Función recursiva que recorre el árbol para un único paciente x
        Por lo general, la primera vez que se ejecuta la función pasaremos la raíz del árbol como primer nodo
        '''
        # Si llegamos a una hoja devolvemos el valor
        if node.is_leaf_node():
            return node.value
        
        # Creamos la lógica de decisión: x[node.feature] accede al valor clínico de un paciente
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left) # Nos vamos al nodo de la izquierda
        
        # Si no nos fuimos para la izquierda, nos vamos para la derecha
        return self._traverse_tree(x, node.right)


if __name__ == '__main__':
    X_test = np.array([[1, 2], [2, 3], [3, 4], [7, 8], [8, 9], [9, 10]])
    y_test = np.array([0, 0, 0, 1, 1, 1])

    tree = DecisionTree(max_depth=3)
    tree.fit(X_test, y_test)
    
    predicciones = tree.predict(np.array([[2, 2], [10, 10]]))
    print(f"Predicciones (deberían ser [0, 1]): {predicciones}")