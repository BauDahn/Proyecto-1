import React, { useState } from 'react';
import FormularioMedico from './components/Formulario';
import { obtenerPrediccion } from './services/api';

function App() {
  const [activeTab, setActiveTab] = useState('formulario'); // 'formulario' o 'evaluacion'
  const [subTabModelo, setSubTabModelo] = useState('logistica'); // 'logistica' o 'arbol'
  const [resultado, setResultado] = useState(null);
  const [cargando, setCargando] = useState(false);
  const [error, setError] = useState(null);

  const manejarEnvio = async (datosPaciente) => {
    setCargando(true);
    setError(null);
    setResultado(null);
    
    try {
      const res = await obtenerPrediccion(datosPaciente);
      setResultado(res);
    } catch (err) {
      setError("No se pudo conectar con el servidor médico. Verifique que el backend esté encendido.");
    } finally {
      setCargando(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans antialiased">
      {/* Encabezado del Sistema */}
      <header className="bg-slate-800 border-b border-slate-700 py-6 px-4 shadow-md w-full">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <h1 className="text-2xl font-black text-white tracking-tight flex items-center gap-2">
              <span>🩺</span> Sistema de Predicción de Riesgo Clínico
            </h1>
            <p className="text-slate-400 text-xs mt-1">Plataforma de IA para soporte en la toma de decisiones médicas</p>
          </div>
          <div className="flex bg-slate-900 p-1 rounded-xl border border-slate-700">
            <button 
              onClick={() => setActiveTab('formulario')}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${activeTab === 'formulario' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
            >
              📊 Diagnóstico Individual
            </button>
            <button 
              onClick={() => setActiveTab('evaluacion')}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${activeTab === 'evaluacion' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
            >
              🔬 Evaluación del Modelo
            </button>
          </div>
        </div>
      </header>

      {/* Contenido Principal */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        {activeTab === 'formulario' ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl">
              <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                <span>📝</span> Datos Clínicos del Paciente
              </h2>
              <FormularioMedico onSubmit={manejarEnvio} cargando={cargando} />
            </div>

            <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
              <div>
                <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                  <span>🎯</span> Resultado del Análisis
                </h2>
                {cargando && (
                  <div className="flex flex-col items-center justify-center py-12">
                    <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-500 mb-4"></div>
                    <p className="text-sm text-slate-400 animate-pulse">Procesando variables logísticas...</p>
                  </div>
                )}
                {error && <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 p-4 rounded-xl text-sm mb-4">{error}</div>}
                {!resultado && !cargando && !error && (
                  <div className="text-center py-12 text-slate-500 text-sm border-2 border-dashed border-slate-700 rounded-xl">
                    Ingrese los datos del paciente y presione calcular para ver la probabilidad estimada de tratamiento.
                  </div>
                )}
                {resultado && (
                  <div className="space-y-6 animate-fadeIn">
                    <div className="text-center bg-slate-900/50 p-6 rounded-xl border border-slate-700">
                      <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Probabilidad Estimada (EVAR)</div>
                      <div className="text-5xl font-black text-indigo-400">{(resultado.probabilidad * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-8 animate-fadeIn">
            {/* Sub-pestañas para los dos modelos */}
            <div className="flex border-b border-slate-700 gap-4">
              <button 
                onClick={() => setSubTabModelo('logistica')}
                className={`pb-3 text-sm font-bold tracking-wide transition-all border-b-2 ${subTabModelo === 'logistica' ? 'border-indigo-500 text-white' : 'border-transparent text-slate-400 hover:text-slate-200'}`}
              >
                📈 Regresión Logística (Modelo Continuo)
              </button>
              <button 
                onClick={() => setSubTabModelo('arbol')}
                className={`pb-3 text-sm font-bold tracking-wide transition-all border-b-2 ${subTabModelo === 'arbol' ? 'border-indigo-500 text-white' : 'border-transparent text-slate-400 hover:text-slate-200'}`}
              >
                🌿 Árbol de Decisión
              </button>
            </div>

            {subTabModelo === 'logistica' ? (
              <div className="space-y-10">
                {/* SECCIÓN 1: MÉTRICA ÚNICA Y CURVA ROC */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-start">
                  {/* Tarjeta de Métrica Única */}
                  <div className="bg-slate-800 border border-slate-700 p-6 rounded-2xl shadow-xl flex flex-col justify-center h-full">
                    <div className="text-xs font-bold text-indigo-400 tracking-wider uppercase mb-1">Métrica de Discriminación</div>
                    <div className="text-5xl font-black text-white my-2">0.707</div>
                    <div className="text-sm font-bold text-slate-300">AUC-ROC (5-Fold Cross-Validation)</div>
                    <p className="text-xs text-slate-400 mt-3 leading-relaxed">
                      Evaluado de forma pura sin umbrales de corte fijos. Esta métrica representa la probabilidad de que el modelo ordene correctamente a un paciente que recibirá reparación endovascular de aneurisma (EVAR, clase 1) por encima de uno asignado a cirugía cardiovascular abierta (clase 0). Al usar validación cruzada <strong>Out-of-Fold</strong>, la estimación está libre de sobreajuste y protege la varianza real del estimador.
                    </p>
                  </div>

                  {/* Imagen de la Curva ROC */}
                  <div className="md:col-span-2 bg-slate-800 border border-slate-700 p-6 rounded-2xl shadow-xl">
                    <h3 className="font-bold text-white text-base mb-3 flex items-center gap-2">
                      <span>📊</span> Curva ROC Global Estabilizada
                    </h3>
                    <div className="bg-slate-900 p-2 rounded-xl border border-slate-700 flex justify-center items-center">
                      <img 
                        src="/curva_roc.png" 
                        alt="Curva ROC de la Validación Cruzada en R" 
                        className="max-h-[300px] object-contain"
                      />
                    </div>
                  </div>
                </div>

                {/* SECCIÓN 2: TRATAMIENTO Y VALIDACIÓN DE DATOS */}
                <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl">
                  <h3 className="font-bold text-white text-base mb-3 flex items-center gap-2">
                    <span>⚙️</span> Trabajo de los datos y validación del modelo
                  </h3>
                  <div className="text-slate-300 text-sm space-y-3 leading-relaxed">
                    <p>
                      Para capturar la varianza real de la muestra clínica, los datos crudos pasaron por un pipeline riguroso antes del análisis estadístico:
                    </p>
                    <ul className="list-disc pl-5 space-y-1 text-slate-400 text-xs">
                      <li><strong className="text-slate-300">Limpieza y Tipado:</strong> Se eliminaron registros incompletos en variables críticas y se parsearon los factores categóricos (<code className="text-indigo-300">Sexo</code>, <code className="text-indigo-300">Fumador</code>, y agrupaciones de comorbilidades) asegurando niveles de referencia clínicamente lógicos.</li>
                      <li><strong className="text-slate-300">Trabajo del modelo:</strong> Se probaron varios modelos diferentes, justificando interacciones de variables en base a investigaciones pasadas. Se terminó encontrando una fuerte relación creciente entre la cantidad de comorbilidades del paciente y el tipo de tratamiento que recibirá.</li>
                      <li><strong className="text-slate-300">Validación Cruzada Estratificada (5-Fold CV):</strong> El dataset completo (<code className="text-slate-200">dataset_clean</code>) se fragmentó de forma aleatoria en 5 bloques o pliegues mutuamente excluyentes, forzando a que cada bloque mantuviera exactamente la proporción original de pacientes de cirugía abierta y EVAR.</li>
                      <li><strong className="text-slate-300">Predicción Out-of-Fold:</strong> La curva ROC superior no se evaluó mediante un hold-out tradicional (que desecharía valiosa varianza médica), sino acumulando secuencialmente las predicciones continuas generadas sobre cada pliegue de prueba mientras este actuaba como un entorno ciego.</li>
                    </ul>
                  </div>
                </div>

                {/* SECCIÓN 3: FÓRMULA MATEMÁTICA DEL MODELO FINAL */}
                <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl">
                  <h3 className="font-bold text-white text-base mb-4 flex items-center gap-2">
                    <span>🧮</span> Ecuación Matemática del Modelo Generalizado
                  </h3>
                  <p className="text-slate-400 text-sm mb-4 leading-relaxed">
                    El modelo calcula la probabilidad continua aplicando la función logística (sigmoide) sobre la combinación lineal de los coeficientes (pesos) extraídos directamente de la optimización por Máxima Verosimilitud en R:
                  </p>
                  <div className="bg-slate-900 p-5 rounded-xl border border-slate-700 font-mono text-center overflow-auto text-indigo-400 text-sm md:text-base">
                    P(EVAR) = 1 / (1 + e^-(-6.4583 + 0.0845·Edad - 0.7148·Sexo + 0.3508·Fumador_Actual + 1.6747·Ex_Fumador + 0.4066·comorb_Medio + 1.3136·comorb_Alto))
                  </div>
                </div>

                {/* SECCIÓN 4: DESGLOSE DE COEFICIENTES UNO A UNO */}
                <div className="space-y-4">
                  <h3 className="font-bold text-white text-base pl-1 flex items-center gap-2">
                    <span>🔍</span> Impacto Clínico y Efecto de los Coeficientes (Pesos de R)
                  </h3>
                  
                  {/* Coeficiente 1: Intercepto */}
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md transition-all hover:border-slate-600">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-mono text-indigo-400 font-bold">β₀: Intercepto (Intercept)</span>
                      <span className="bg-slate-900 px-3 py-1 text-xs font-mono rounded-md text-slate-300 font-bold">-6.4583</span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Representa el log-odds basal de requerir EVAR cuando todas las variables predictoras continuas son cero y las categóricas están en su nivel de referencia (Paciente Masculino, No Fumador, y con pocas comorbilidades: grupo 0-3). Su valor fuertemente negativo indica que el riesgo inicial de derivación a EVAR para este perfil base es extremadamente bajo, inclinando la balanza hacia la cirugía abierta de control.
                    </p>
                  </div>

                  {/* Coeficiente 2: Edad */}
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md transition-all hover:border-slate-600">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-mono text-indigo-400 font-bold">β₁: Edad (Variable Continua)</span>
                      <span className="bg-slate-900 px-3 py-1 text-xs font-mono rounded-md text-emerald-400 font-bold">+0.0845</span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Indica el cambio en el log-odds de asignación a EVAR por cada año adicional de vida del paciente. Al ser positivo, refleja que a mayor edad, la probabilidad de optar por el tratamiento endovascular (EVAR) se incrementa de forma exponencial suave. Clínicamente se justifica porque el envejecimiento eleva el riesgo quirúrgico global, desaconsejando la agresividad de una cirugía abierta.
                    </p>
                  </div>

                  {/* Coeficiente 3: Sexo */}
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md transition-all hover:border-slate-600">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-mono text-indigo-400 font-bold">β₂: Sexo (Factor: 0 = Male, 1 = Female)</span>
                      <span className="bg-slate-900 px-3 py-1 text-xs font-mono rounded-md text-rose-400 font-bold">-0.7148</span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Mide el impacto de ser paciente femenino (<code className="text-slate-300">Sexo = 1</code>) tomando como nivel base a los hombres. Al tener un peso negativo sustancial, el modelo indica que el hecho de ser mujer reduce de forma significativa los odds de recibir un EVAR en comparación con un hombre bajo las mismas condiciones clínicas, lo cual se alinea con criterios anatómicos y de distribución epidemiológica de los aneurismas.
                    </p>
                  </div>

                  {/* Coeficiente 4: Fumador Actual */}
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md transition-all hover:border-slate-600">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-mono text-indigo-400 font-bold">β₃: Tabaquismo (Fumador Actual vs. No Fumador)</span>
                      <span className="bg-slate-900 px-3 py-1 text-xs font-mono rounded-md text-emerald-400 font-bold">+0.3508</span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Representa el incremento en el log-odds para pacientes que fuman activamente en comparación con el grupo base de control (No fumadores). Su impacto positivo refleja que el tabaquismo activo desplaza la probabilidad hacia la recomendación de EVAR, vinculándose directamente con un mayor deterioro y fragilidad del tejido arterial elastoplástico circundante.
                    </p>
                  </div>

                  {/* Coeficiente 5: Ex-Fumador */}
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md transition-all hover:border-slate-600">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-mono text-indigo-400 font-bold">β₄: Tabaquismo (Ex-Fumador vs. No Fumador)</span>
                      <span className="bg-slate-900 px-3 py-1 text-xs font-mono rounded-md text-emerald-400 font-bold">+1.6747</span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Este es uno de los coeficientes con mayor magnitud del modelo. Evalúa el impacto de los pacientes ex-fumadores frente a los no fumadores. Un valor positivo tan alto ($+1.6747$) sugiere que el historial de tabaquismo crónico severo en el pasado deja secuelas estructurales permanentes o se asocia fuertemente en esta muestra con perfiles de alto riesgo anatómico/sistémico, haciendo que el modelo dispare la probabilidad hacia la intervención protectora endovascular (EVAR).
                    </p>
                  </div>

                  {/* Coeficiente 6: Comorbilidades Grupo Medio */}
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md transition-all hover:border-slate-600">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-mono text-indigo-400 font-bold">β₅: Comorbilidades Grupo Medio (3 a 6 patologías vs. Grupo 0-3)</span>
                      <span className="bg-slate-900 px-3 py-1 text-xs font-mono rounded-md text-emerald-400 font-bold">+0.4066</span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Mide el efecto de pasar de una carga comórbida baja (0-3 patologías) a una moderada (3-6 patologías). Al ser un coeficiente positivo, el modelo empieza a penalizar el estado general del paciente, aumentando de manera clara el log-odds a favor de EVAR, debido a que el riesgo de colapso intraoperatorio en una cirugía abierta convencional empieza a incrementarse.
                    </p>
                  </div>

                  {/* Coeficiente 7: Comorbilidades Grupo Alto */}
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md transition-all hover:border-slate-600">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-mono text-indigo-400 font-bold">β₆: Comorbilidades Grupo Alto (6 a 9 patologías vs. Grupo 0-3)</span>
                      <span className="bg-slate-900 px-3 py-1 text-xs font-mono rounded-md text-emerald-400 font-bold">+1.3136</span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Representa el impacto crítico de pacientes con una carga patológica severa (6-9 comorbilidades concurrentes). Su valor fuertemente positivo (+1.3136) confirma la hipótesis principal del modelo de IA: ante un panorama clínico tan comprometido, la probabilidad se desplaza de manera masiva y contundente alejándose de la cirugía abierta (0) y consolidando al EVAR (1) como la única alternativa terapéutica endovascular viable y segura para preservar la vida del paciente.
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              /* SECCIÓN DETALLADA DEL ÁRBOL DE DECISIÓN (Sincronizada del Código 2) */
              <div className="space-y-6 animate-fadeIn">
                <div className="bg-slate-800 border border-slate-700 p-6 rounded-2xl shadow-xl">
                  <h2 className="text-xl font-black text-white mb-2">Métricas del Árbol de Decisión Clínico</h2>
                  <p className="text-slate-400 text-sm leading-relaxed">
                    Evaluación del modelo de clasificación no paramétrico basado en reglas jerárquicas lógicas y la ganancia de información (Entropía/Gini).
                  </p>
                </div>

                {/* Cuadrícula de Métricas del Árbol */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md">
                    <div className="text-xs font-bold text-indigo-400 tracking-wider uppercase mb-1">Accuracy</div>
                    <div className="text-3xl font-black text-white">62.2%</div>
                    <div className="text-xs text-slate-500 mt-2">Accuracy obtenido usando un umbral de 0.5</div>
                  </div>
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md">
                    <div className="text-xs font-bold text-teal-400 tracking-wider uppercase mb-1">Sensibilidad</div>
                    <div className="text-3xl font-black text-white">62%</div>
                    <div className="text-xs text-slate-500 mt-2">Capacidad de captación de positivos verdaderos</div>
                  </div>
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md">
                    <div className="text-xs font-bold text-amber-400 tracking-wider uppercase mb-1">Profundidad del Árbol</div>
                    <div className="text-3xl font-black text-white">4 Niveles</div>
                    <div className="text-xs text-slate-500 mt-2">Complejidad controlada mediante poda (Pruning)</div>
                  </div>
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md">
                    <div className="text-xs font-bold text-rose-400 tracking-wider uppercase mb-1">Nodos Terminales</div>
                    <div className="text-3xl font-black text-white">8 Hojas</div>
                    <div className="text-xs text-slate-500 mt-2">Segmentos clínicos finales bien definidos</div>
                  </div>
                </div>

                {/* Resumen del Enfoque del Árbol */}
                <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl">
                  <h3 className="font-bold text-white text-base mb-2">📋 Lógica Jerárquica del Árbol</h3>
                  <p className="text-slate-400 text-sm mb-4 leading-relaxed">
                    A diferencia de la regresión, este modelo opera dividiendo al dataset secuencialmente. La primera división o **Nodo Raíz** evalúa la variable con mayor ganancia de información, facilitando la toma de decisiones médicas mediante un mapa de flujo visual.
                  </p>
                  <div className="bg-slate-900 p-4 rounded-xl border border-slate-700 overflow-auto">
                    <img
                      src="/plots/diagrama_arbol.png"
                      alt="Diagrama del árbol de decisión"
                      className="w-full object-contain"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;