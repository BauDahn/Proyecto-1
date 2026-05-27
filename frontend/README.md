# Frontend: Interfaz de Predicción Vascular

Este directorio contiene la aplicación cliente del proyecto, diseñada para ser utilizada por el personal médico. 

## Stack Tecnológico
La aplicación está construida utilizando herramientas modernas para garantizar un rendimiento óptimo y un desarrollo ágil:
* **React (v19)**: Biblioteca principal para la construcción de interfaces de usuario.
* **Vite (v8)**: Entorno de desarrollo y empaquetador ultrarrápido.
* **Tailwind CSS (v4)**: Framework de utilidades CSS para un diseño responsivo y moderno.

## Características Principales
* **Formulario**: Interfaz para capturar variables de entrada como edad, sexo, si es fumador y comorbilidades (HTA, IAM, ERC, etc.).
* **Visualización de Explicabilidad**: El frontend renderiza de forma dinámica las visualizaciones generadas por el modelo predictivo del backend. Concretamente, el backend guarda gráficos estructurales (como `diagrama_arbol.png`) directamente en la carpeta `/public/plots` de este entorno para que React pueda mostrarlos al usuario de forma instantánea.

## Instalación y Ejecución Local
Para arrancar el entorno de desarrollo local:
1.  Instala las dependencias: `npm install`
2.  Inicia el servidor de Vite: `npm run dev`