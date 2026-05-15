// Conexión con el backend
const API_URL = "http://127.0.0.1:8000";

export const obtenerPrediccion = async (datos) => {
  try {
    const response = await fetch(`${API_URL}/predictor`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(datos),
    });

    if (!response.ok) {
      throw new Error('Error en la API');
    }

    return await response.json();
  } catch (error) {
    console.error("Error de conexión:", error);
    throw error;
  }
};