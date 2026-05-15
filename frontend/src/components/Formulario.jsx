import React, { useState } from 'react';

const FormularioMedico = ({ onSubmit }) => {
  const [formData, setFormData] = useState({
    edad: 25, 
    sexo: false, 
    fumador: 0, 
    dislipemia: false,
    diabetes: false, 
    hta: false, 
    iam: false, 
    erc: false,
    epoc: false, 
    acv: false, 
    fa: false, 
    cancer: false, // ¡Cambiado de ancer a cancer!
    icc: false, 
    eap: false
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    let finalValue;

    if (type === 'checkbox') {
      finalValue = checked;
    } else if (name === 'edad') {
      // Permite borrar el número sin que explote
      finalValue = value === '' ? '' : parseInt(value);
    } else if (name === 'fumador') {
      finalValue = parseInt(value);
    } else {
      finalValue = value;
    }

    setFormData(prev => ({ ...prev, [name]: finalValue }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Aseguramos que la edad sea un número antes de enviar
    const datosEnvio = {
      ...formData,
      edad: formData.edad === '' ? 0 : Number(formData.edad)
    };
    onSubmit(datosEnvio);
  };

  return (
    <div className="max-w-4xl mx-auto my-10 p-8 bg-white shadow-2xl rounded-2xl border border-gray-100">
      <div className="text-center mb-10">
        <h2 className="text-3xl font-extrabold text-slate-800">Ficha Clínica del Paciente</h2>
        <p className="text-slate-500 mt-2 font-medium">Complete los datos demográficos y antecedentes</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Sección 1: Datos Demográficos */}
        <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
          <h3 className="text-lg font-semibold text-indigo-700 mb-4">👤 Datos Demográficos</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-left">
            <div>
              <label className="block text-sm font-bold text-slate-700 mb-1">Edad (años)</label>
              <input 
                type="number" 
                name="edad" 
                value={formData.edad} 
                onChange={handleChange}
                className="block w-full rounded-lg border-slate-300 text-slate-900 font-bold shadow-sm focus:border-indigo-500 focus:ring-indigo-500 p-2.5 border bg-white" 
              />
            </div>

            <div>
              <label className="block text-sm font-bold text-slate-700 mb-1">Estado Tabáquico</label>
              <select 
                name="fumador" 
                value={formData.fumador} 
                onChange={handleChange}
                className="block w-full rounded-lg border-slate-300 text-slate-900 font-bold shadow-sm focus:border-indigo-500 focus:ring-indigo-500 p-2.5 border bg-white"
              >
                <option value={0}>🚫 No fuma</option>
                <option value={1}>🚬 Fumador activo</option>
                <option value={2}>⏱️ Ex-fumador</option>
              </select>
            </div>

            <div className="flex flex-col">
              <span className="block text-sm font-bold text-slate-700 mb-1">Sexo</span>
              <label className="relative inline-flex items-center cursor-pointer mt-2">
                <input type="checkbox" name="sexo" checked={formData.sexo} onChange={handleChange} className="sr-only peer" />
                <div className="w-11 h-6 bg-slate-300 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-indigo-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
                <span className="ml-3 text-sm font-bold text-slate-700">{formData.sexo ? 'Masculino' : 'Femenino'}</span>
              </label>
            </div>
          </div>
        </div>

        {/* Sección 2: Antecedentes */}
        <div className="text-left">
          <h3 className="text-lg font-semibold text-indigo-700 mb-4">📋 Antecedentes Patológicos</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
            {Object.keys(formData).map((key) => {
              if (['edad', 'sexo', 'fumador'].includes(key)) return null;
              return (
                <label key={key} className={`flex items-center p-3 border rounded-xl cursor-pointer transition-all ${formData[key] ? 'bg-indigo-50 border-indigo-300 shadow-sm' : 'bg-white border-slate-200 hover:border-indigo-200'}`}>
                  <input 
                    type="checkbox" 
                    name={key} 
                    checked={formData[key]} 
                    onChange={handleChange}
                    className="h-5 w-5 text-indigo-600 focus:ring-indigo-500 border-slate-300 rounded" 
                  />
                  <span className="ml-3 text-xs font-black text-slate-800 uppercase tracking-tight">
                    {key}
                  </span>
                </label>
              );
            })}
          </div>
        </div>

        <div className="flex gap-4 pt-4">
          <button type="submit" className="flex-1 bg-indigo-600 text-white py-4 px-6 rounded-2xl font-black text-lg hover:bg-indigo-700 transition-all shadow-lg active:scale-95">
            Calcular Riesgo Predictivo
          </button>
          <button 
            type="button" 
            onClick={() => setFormData({edad:25, sexo:false, fumador:0, dislipemia:false, diabetes:false, hta:false, iam:false, erc:false, epoc:false, acv:false, fa:false, cancer:false, icc:false, eap:false})}
            className="bg-slate-200 text-slate-700 py-4 px-8 rounded-2xl font-bold hover:bg-slate-300 transition-all"
          >
            Limpiar
          </button>
        </div>
      </form>
    </div>
  );
};

export default FormularioMedico;