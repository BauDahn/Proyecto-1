#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // Convierte tuplas de std a python
#include <tuple>
#include <vector>
#include <algorithm>
#include <map>

namespace py = pybind11;

double _calcular_gini_interno(const py::detail::unchecked_reference<int, 1>& copia) {
    /*
    Esta función trabaja solo con C++ para facilitar el cálculo del gini sin tener que mirar el código de python.
    Recibe como parámetros:
    Una constante de un array generado por pybind que tiene definido dónde empieza y termina en la ram que tiene enteros y es unidimensional, y además se recibe como una referencia (&)
    El map es un diccionario con la frecuencia de cada etiqueta y en el nodo que se crea en ese split.
    cantidad_pacientes = número de pacientes dentro del nodo
    copia(i) = etiqueta del paciente i
    impureza = impureza gini
    */
    int cantidad_pacientes = copia.shape(0);
    if (cantidad_pacientes == 0) return 0.0;

    std::map<int, int> diccionario_frecuencias;
    for (int i = 0; i < cantidad_pacientes; i++) {
        diccionario_frecuencias[copia(i)]++;
    }

    double impureza = 1.0; // Arrancamos la impureza desde el 1
    for (auto const& [etiqueta, cantidad] : diccionario_frecuencias) {
        double probabilidad = static_cast<double>(cantidad) / cantidad_pacientes;
        impureza -= probabilidad * probabilidad;
    }
    return impureza;
    
}

double calcular_gini(py::array_t<int> y) {
    /*
    La función recibe el array de numpy con las etiquetas posibles de la variable y
    Accedemos a el array de una sola dimensión de numpy sin chekear si salimos de los límites del array
    */
    auto copia = y.unchecked<1>();
    return _calcular_gini_interno(copia);
}


std::tuple<int, double> best_split(py::array_t<double> X, py::array_t<int> y, int cantidad_minima_en_nodo) { // array_t forma parte de pybind 11 y refiere a array de numpy
    auto X_ref = X.unchecked<2>(); // matriz
    auto y_ref = y.unchecked<1>(); // vector

    int cantidad_pacientes = X_ref.shape(0); // Indica el número de filas
    int cantidad_features = X_ref.shape(1); // Número de variables clínicas

    int mejor_feature = -1;
    double mejor_umbral = -1.0;
    double mejor_ganancia = 0.0;

    double gini_padre = _calcular_gini_interno(y_ref);

    for (int feature = 0 ; feature < cantidad_features ; feature++) {
        // Primero crearé un vector <pair<double, int>> para almacenar el dato y la clase
        // El tamaño va a ser el número de muestras
        std::vector<std::pair<double, int>> muestras;
        muestras.reserve(cantidad_pacientes);

        // Llenamos el vector de datos
        for (int i = 0 ; i < cantidad_pacientes ; i++) {
            // X_ref(i, columna) es el valor de la característica, y_ref(i) es la clasificación
            muestras.push_back({X_ref(i, feature), y_ref(i)});
        }

        // Ahora hacemos el ordenamiento del vector de datos, para buscar el mejor umbral
        std::sort(muestras.begin(), muestras.end());

        // Una vez sorteado el array de pares (que se sortea siempre según el valor del feature), pasamos a recorrer el array buscando el mejor gini
        // Voy a usar una ventana deslizante
        std::map<int, int> nodo_izq, nodo_der; // Creo dos diccionarios, uno para los de la izquierda y otros a la derecha
        for (auto const& p : muestras) { // Por cada par en muestras, los sumo todos en el total de la derecha
            nodo_der[p.second]++; // O le sumo 1 a la clase 0, 1 o 2.
        }

        int total_izq = 0;
        int total_der = cantidad_pacientes;
        
        // Recorro el array buscando el corte:
        for (int i = 0 ; i < cantidad_pacientes - 1 ; i++) {
            // Muevo al paciente i de derecha a izquierda
            int etiqueta = muestras[i].second;
            nodo_izq[etiqueta]++;
            nodo_der[etiqueta]--;
            total_izq++;
            total_der--;

            // Hay que evitar hacer cortes de valores iguales
            if (muestras[i].first == muestras[i + 1].first) continue;

            // Si el corte deja muy pocos pacientes de un lado lo ignoramos
            if (total_izq < cantidad_minima_en_nodo || total_der < cantidad_minima_en_nodo) continue;

            // Cálculo del gini a ambos lados
            double gini_izq = 1.0, gini_der = 1.0;
            for (auto const& [et, cant] : nodo_izq) {
                double p = static_cast<double>(cant) / total_izq;
                gini_izq -= p * p;
            }
            for (auto const& [et, cant] : nodo_der) {
                double p = static_cast<double>(cant) / total_der;
                gini_der -= p * p;
            }

            // Cálculo de la ganancia luego del corte
            double gini_medio = (static_cast<double>(total_izq) / cantidad_pacientes) * gini_izq + 
                                (static_cast<double>(total_der) / cantidad_pacientes) * gini_der;
            double ganancia = gini_padre - gini_medio;

            // Miramos si es el mejor corte
            if (ganancia > mejor_ganancia) {
                mejor_ganancia = ganancia;
                mejor_feature = feature;
                mejor_umbral = (muestras[i].first + muestras[i+1].first) / 2.0; // Calculo del punto medio
            }
        }

    }
    return std::make_tuple(mejor_feature, mejor_umbral);
}

// Documentación de la función para Python.
PYBIND11_MODULE(tree_core, m) {
    m.doc() = "Motor optimizado para el Proyecto Aneurismas";

    m.def("calcular_gini", &calcular_gini, 
        "Calcula la impureza de Gini de un array de NumPy (int32).");
    
    m.def("encontrar_mejor_corte", &best_split, py::arg("X"), py::arg("y"), py::arg("cantidad_minima_en_nodo") = 1);
}