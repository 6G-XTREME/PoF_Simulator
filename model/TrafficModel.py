import numpy as np

def f_gaussiana_base(t, _sigma=4*3600, _mu=12*3600):
    return 1 - np.exp(-((t - _mu)**2) / (2 * _sigma**2))

def f_gaussiana_pico(t, _sigma=1*3600, _mu=20*3600):
    return np.exp(-0.5 * ((t - _mu - 6*3600) / _sigma) ** 2)

def f_gaussiana_finde_pulso(t, center=6.5*24*3600, transition_width=3600, width=2*24*3600, _limit=0.4):
    """
    Pulso centrado correctamente: el centro es el punto medio del pulso.
    """
    def sigmoid(x, midpoint, slope):
        return 1 / (1 + np.exp(-(x - midpoint) / slope))
    
    # El pulso va de (center - width/2) a (center + width/2)
    start = center - width / 2
    end = center + width / 2
    rise = sigmoid(t, start, transition_width)
    fall = sigmoid(t, end, transition_width)
    return 1 - _limit * (rise - fall)

# Función modificada con crecimiento desde 10am y pico en 8pm
def valley_peak_traffic_profile(t_segundos, _alpha=0.2, _offset=0.2):
    """
    Devuelve el perfil de tráfico evaluado para tiempo en segundos.
    """
    # Función normal complementaria
    return _offset + (1 - _offset) * (
        f_gaussiana_base(t_segundos)
    )

def estimate_traffic_from_seconds(t_segundos, ruido=True, ruido_max=0.015, seed=None):
    """
    Devuelve el factor de tráfico para un instante temporal dado en segundos.
    Usa interpolación con resolución de 15 minutos (900s) y ruido opcional.
    """
    # Desplazamos el tiempo 6 horas hacia adelante, para ajustar el valle de la función
    t_segundos = t_segundos + 6 * 3600

    # Normalizar el tiempo al rango de un día
    t_segundos_norm = t_segundos % (24 * 3600)  # segundos en un día

    # Crear vector base de referencia (cada 15 minutos = 900s)
    tiempos_base = np.arange(0, 24 * 3600, 900)
    perfil_base = valley_peak_traffic_profile(tiempos_base)

    # Interpolación
    valor_estimado = np.interp(t_segundos_norm, tiempos_base, perfil_base)

    # Aplicar ruido
    if ruido:
        if seed is not None:
            np.random.seed(seed)
        delta = np.random.uniform(-ruido_max, ruido_max)
        valor_estimado += delta

    # Aplicar el pulso de fin de semana
    valor_estimado *= f_gaussiana_finde_pulso(t_segundos)

    # return np.clip(valor_estimado, 0.1, 1.0)
    return valor_estimado