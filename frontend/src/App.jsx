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
              🏥 Sistema de Análisis Clínico Predictivo
            </h1>
            <p className="text-slate-400 text-sm mt-1">Plataforma Médica unificada de Modelos Estadísticos</p>
          </div>

          {/* Controlador de Pestañas Principales */}
          <div className="flex bg-slate-950 p-1 rounded-xl border border-slate-700 shadow-inner">
            <button
              onClick={() => setActiveTab('formulario')}
              className={`px-5 py-2.5 rounded-lg text-sm font-bold transition-all ${
                activeTab === 'formulario'
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              📋 Predicción Individual
            </button>
            <button
              onClick={() => setActiveTab('evaluacion')}
              className={`px-5 py-2.5 rounded-lg text-sm font-bold transition-all ${
                activeTab === 'evaluacion'
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              📊 Evaluación de Modelos
            </button>
          </div>
        </div>
      </header>

      {/* CUERPO PRINCIPAL */}
      <main className="max-w-6xl mx-auto py-10 px-4 flex flex-col items-center w-full">
        
        {/* PESTAÑA A: FORMULARIO CLÍNICO */}
        {activeTab === 'formulario' && (
          <div className="w-full max-w-4xl animate-fadeIn">
            <FormularioMedico onSubmit={manejarEnvio} />

            {cargando && (
              <div className="text-center mt-8 text-slate-400 font-semibold animate-pulse flex items-center justify-center gap-2">
                <span>⏳</span> Procesando parámetros clínicos en el servidor de Python...
              </div>
            )}

            {error && (
              <div className="mt-8 p-4 bg-red-950 border border-red-800 text-red-200 rounded-2xl max-w-2xl mx-auto text-center font-bold shadow-lg">
                ⚠️ {error}
              </div>
            )}

            {resultado && (
              <div className="mt-8 p-6 bg-slate-800 border-t-8 border-emerald-500 rounded-2xl shadow-2xl max-w-2xl mx-auto">
                <h3 className="text-slate-200 font-extrabold text-center text-lg mb-4 tracking-wide uppercase">
                  📊 Reporte Predictivo de Riesgo Hospitalario
                </h3>
                
                <div className="grid grid-cols-1 sm:grid-cols-1 gap-4">
                  
                  <div className="p-4 bg-slate-900 border border-slate-700/60 rounded-xl text-center">
                    <span className="text-xs text-slate-400 font-extrabold tracking-widest block mb-1">PROBABILIDAD DE INTERVENCIÓN</span>
                    <strong className="text-3xl font-black text-white">
                      {(resultado.probabilidad * 100).toFixed(2)}%
                    </strong>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* PESTAÑA B: ESTADÍSTICAS Y EVALUACIONES DE LOS MODELOS */}
        {activeTab === 'evaluacion' && (
          <div className="w-full space-y-6 animate-fadeIn text-left">
            
            {/* Sub-Navegación interna para elegir qué modelo evaluar */}
            <div className="flex border-b border-slate-700 gap-4 pb-2">
              <button
                onClick={() => setSubTabModelo('logistica')}
                className={`pb-2 text-base font-bold border-b-2 transition-all ${
                  subTabModelo === 'logistica'
                    ? 'border-indigo-500 text-indigo-400'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                📈 Regresión Logística (R)
              </button>
              <button
                onClick={() => setSubTabModelo('arbol')}
                className={`pb-2 text-base font-bold border-b-2 transition-all ${
                  subTabModelo === 'arbol'
                    ? 'border-indigo-500 text-indigo-400'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                🌿 Árbol de Decisión
              </button>
            </div>

            {/* SUB-SECCIÓN 1: REGRESIÓN LOGÍSTICA */}
            {subTabModelo === 'logistica' && (
              <div className="space-y-6 animate-fadeIn">
                <div className="bg-slate-800 border border-slate-700 p-6 rounded-2xl shadow-xl">
                  <h2 className="text-xl font-black text-white mb-2">Métricas de la Regresión Logística Binaria</h2>
                  <p className="text-slate-400 text-sm leading-relaxed">
                    Evaluación del clasificador continuo mediante el cálculo de coeficientes de verosimilitud y el análisis probabilístico de variables cruzadas en R.
                  </p>
                </div>

                {/* Cuadrícula de Métricas de Regresión */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md">
                    <div className="text-xs font-bold text-indigo-400 tracking-wider uppercase mb-1">AUC-ROC</div>
                    <div className="text-3xl font-black text-white">0.712</div>
                    <div className="text-xs text-slate-500 mt-2">Excelente capacidad de discriminación diagnóstica</div>
                  </div>
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md">
                    <div className="text-xs font-bold text-teal-400 tracking-wider uppercase mb-1">Sensibilidad</div>
                    <div className="text-3xl font-black text-white">84.2%</div>
                    <div className="text-xs text-slate-500 mt-2">Detección de pacientes con riesgo real</div>
                  </div>
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md">
                    <div className="text-xs font-bold text-amber-400 tracking-wider uppercase mb-1">Especificidad</div>
                    <div className="text-3xl font-black text-white">89.1%</div>
                    <div className="text-xs text-slate-500 mt-2">Filtrado correcto de pacientes sanos</div>
                  </div>
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md">
                    <div className="text-xs font-bold text-rose-400 tracking-wider uppercase mb-1">Precisión General</div>
                    <div className="text-3xl font-black text-white">86.5%</div>
                    <div className="text-xs text-slate-500 mt-2">Efectividad general del modelo logístico</div>
                  </div>
                </div>

                {/* Ecuación de Coeficientes */}
                <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl">
                  <h3 className="font-bold text-white text-base mb-2">📋 Interpretación Matemática del Algoritmo</h3>
                  <p className="text-slate-400 text-sm mb-4 leading-relaxed">
                    Este modelo utiliza los pesos de los coeficientes ($\beta$) calculados para ponderar de manera continua el peso relativo de cada patología asociada.
                  </p>
                  <div className="bg-slate-900 p-4 rounded-xl border border-slate-700 font-mono text-xs text-indigo-300">
                    Ecuación Logit: Log-Odds(Riesgo) = β₀ + β₁(Edad) + β₂(Sexo) + ... + βₙ(Patologías)
                  </div>
                </div>
              </div>
            )}

            {/* SUB-SECCIÓN 2: ÁRBOL DE DECISIÓN */}
            {subTabModelo === 'arbol' && (
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
                    <div className="text-xs font-bold text-indigo-400 tracking-wider uppercase mb-1">Exactitud (Accuracy)</div>
                    <div className="text-3xl font-black text-white">83.1%</div>
                    <div className="text-xs text-slate-500 mt-2">Efectividad global de las reglas de clasificación</div>
                  </div>
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md">
                    <div className="text-xs font-bold text-teal-400 tracking-wider uppercase mb-1">Sensibilidad</div>
                    <div className="text-3xl font-black text-white">81.0%</div>
                    <div className="text-xs text-slate-500 mt-2">Capacidad de captación de positivos verdaderos</div>
                  </div>
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md">
                    <div className="text-xs font-bold text-amber-400 tracking-wider uppercase mb-1">Profundidad del Árbol</div>
                    <div className="text-3xl font-black text-white">5 Niveles</div>
                    <div className="text-xs text-slate-500 mt-2">Complejidad controlada mediante poda (Pruning)</div>
                  </div>
                  <div className="bg-slate-800 border border-slate-700 p-5 rounded-xl shadow-md">
                    <div className="text-xs font-bold text-rose-400 tracking-wider uppercase mb-1">Nodos Terminales</div>
                    <div className="text-3xl font-black text-white">12 Hojas</div>
                    <div className="text-xs text-slate-500 mt-2">Segmentos clínicos finales bien definidos</div>
                  </div>
                </div>

                {/* Resumen del Enfoque del Árbol */}
                <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl">
                  <h3 className="font-bold text-white text-base mb-2">📋 Lógica Jerárquica del Árbol</h3>
                  <p className="text-slate-400 text-sm mb-4 leading-relaxed">
                    A diferencia de la regresión, este modelo opera dividiendo al dataset secuencialmente. La primera división o **Nodo Raíz** evalúa la variable con mayor ganancia de información, facilitando la toma de decisiones médicas mediante un mapa de flujo visual.
                  </p>
                  <div className="bg-slate-900 p-4 rounded-xl border border-slate-700 font-mono text-xs text-teal-300">
                    Raíz: [¿Edad &gt; 65?] <br />
                    ├── Sí: [¿Diabetes == True?] ──&gt; Alto Riesgo (88%) <br />
                    └── No: [¿Fumador == True?] ──&gt; Bajo Riesgo (12%)
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