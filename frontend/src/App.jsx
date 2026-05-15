import { useState } from 'react';
import Formulario from './components/Formulario';
import { obtenerPrediccion } from './services/api';

function App() {
  const [resultado, setResultado] = useState(null);
  const [cargando, setCargando] = useState(false);

  const manejarEnvio = async (datos) => {
    setCargando(true);
    setResultado(null); // Limpiamos resultado anterior
    try {
      const data = await obtenerPrediccion(datos);
      setResultado(data); // Guardamos la respuesta del backend
    } catch (error) {
      alert("Error al conectar con el backend");
    } finally {
      setCargando(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 py-12 px-4">
      <Formulario onSubmit={manejarEnvio} />

      {/* AQUÍ SE MUESTRA EL RESULTADO DE PREDICT */}
      {resultado && (
        <div className="max-w-4xl mx-auto mt-8 p-6 bg-white rounded-2xl shadow-2xl border-l-8 border-indigo-600 animate-bounce-short">
          <h3 className="text-xl font-black text-slate-800 mb-2">📊 Resultado del Análisis:</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-indigo-50 rounded-lg">
              <p className="text-sm text-indigo-600 font-bold uppercase">Nivel de Riesgo</p>
              <p className="text-3xl font-black text-slate-900">{resultado.riesgo}</p>
            </div>
            <div className="p-4 bg-slate-50 rounded-lg">
              <p className="text-sm text-slate-500 font-bold uppercase">Probabilidad</p>
              <p className="text-3xl font-black text-slate-900">{(resultado.probabilidad * 100).toFixed(1)}%</p>
            </div>
          </div>
          {resultado.nota && (
            <p className="mt-4 text-slate-600 italic border-t pt-4">
              📌 {resultado.nota}
            </p>
          )}
        </div>
      )}

      {cargando && (
        <div className="text-center mt-4 text-white font-bold animate-pulse">
          Procesando datos médicos...
        </div>
      )}
    </div>
  );
}

export default App;