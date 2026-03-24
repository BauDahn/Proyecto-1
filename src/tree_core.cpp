#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <map>

namespace py = pybind11;

double calcular_gini(py::array_t<int> y) {
    /*
    La función recibe el array de numpy con las etiquetas posibles de la variable y
    Accedemos a el array de una sola dimensión de numpy sin chekear si salimos de los límites del array
    El map es un diccionario con la frecuencia de cada etiqueta y en el nodo que se crea en ese split.
    cantidad_pacientes = número de pacientes dentro del nodo
    copia(i) = etiqueta del paciente i
    impureza = impureza gini
    */
    auto copia = y.unchecked<1>();
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

// Documentación de la función para Python.
PYBIND11_MODULE(tree_core, m) {
    m.doc() = "Motor optimizado para el Proyecto Aneurismas";

    m.def("calcular_gini", &calcular_gini, 
        "Calcula la impureza de Gini de un array de NumPy (int32).");
}