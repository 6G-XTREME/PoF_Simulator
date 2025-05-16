import numpy as np

def f_gaussiana_base(t, _sigma=4*3600, _mu=12*3600):
    return 1 - np.exp(-((t - _mu)**2) / (2 * _sigma**2))

def f_gaussiana_pico(t, _sigma=1*3600, _mu=20*3600):
    return np.exp(-0.5 * ((t - _mu - 6*3600) / _sigma) ** 2)


# Precalcular constantes fuera de la función para optimización


_SEMANA_SEGUNDOS = 7 * 24 * 3600

# Parámetros configurables para el pulso de fin de semana
# Día y hora de inicio del pulso (por defecto: día 0, 06:00)
DIA_SUBIDA = 0
HORA_INICIO_SUBIDA = 6
HORA_FIN_SUBIDA = 12
# Día y hora de fin del pulso (por defecto: día 5, 12:00)
DIA_BAJADA = 5
HORA_INICIO_BAJADA = 6
HORA_FIN_BAJADA = 12


# Cálculo de los instantes clave en segundos
_SUBIDA_START = DIA_SUBIDA * 24 * 3600 + HORA_INICIO_SUBIDA * 3600
_SUBIDA_END = DIA_SUBIDA * 24 * 3600 + HORA_FIN_SUBIDA * 3600
_SUBIDA_MID = (_SUBIDA_START + _SUBIDA_END) / 2

_BAJADA_START = DIA_BAJADA * 24 * 3600 + HORA_INICIO_BAJADA * 3600
_BAJADA_END = DIA_BAJADA * 24 * 3600 + HORA_FIN_BAJADA * 3600
_BAJADA_MID = (_BAJADA_START + _BAJADA_END) / 2

def _sigmoid(x, midpoint, width):
    # width controla la "pendiente" de la sigmoide (más pequeño = más abrupto)
    return 1 / (1 + np.exp(-(x - midpoint) / width))

def f_gaussiana_finde_pulso(t_segundos, transition_width=3600, base=0.6, peak=1.0):
    """
    Perfil de pulso para fin de semana:
    - Entre las 06:00 del día 1 y las 12:00 del día 1 ocurre la subida (sigmoide) de base a peak.
    - Entre las 06:00 del día 6 (sábado) y las 12:00 del día 6 ocurre la bajada (sigmoide) de peak a base.
    Las transiciones son suaves (sigmoides).
    """
    # Recortar t_segundos a la semana y convertir a array para operaciones vectorizadas
    t = np.asarray(t_segundos) % _SEMANA_SEGUNDOS

    # Sigmoide de subida: pasa de base a peak
    subida = base + (peak - base) * _sigmoid(t, _SUBIDA_MID, transition_width)
    # Sigmoide de bajada: pasa de peak a base
    bajada = peak - (peak - base) * _sigmoid(t, _BAJADA_MID, transition_width)

    # El perfil es: base fuera del pulso, peak dentro del pulso (subida y bajada suaves)
    # El valor final es el mínimo entre subida y bajada (para formar el "pulso" de fin de semana)
    return np.minimum(subida, bajada)

# def f_gaussiana_finde_pulso(t, center=6.5*24*3600, transition_width=3600, width=2*24*3600, _limit=0.4):
#     """
#     Pulso centrado correctamente: el centro es el punto medio del pulso.
#     """
#     def sigmoid(x, midpoint, slope):
#         return 1 / (1 + np.exp(-(x - midpoint) / slope))
    
#     # El pulso va de (center - width/2) a (center + width/2)
#     start = center - width / 2
#     end = center + width / 2
#     rise = sigmoid(t, start, transition_width)
#     fall = sigmoid(t, end, transition_width)
#     return 1 - _limit * (rise - fall)

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
    t_segundos_des = t_segundos + 6 * 3600

    # Normalizar el tiempo al rango de un día
    t_segundos_norm = t_segundos_des % (24 * 3600)  # segundos en un día

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
    t_segundos_semana = t_segundos_des % (7 * 24 * 3600)
    valor_estimado *= f_gaussiana_finde_pulso(t_segundos_semana)

    # return np.clip(valor_estimado, 0.1, 1.0)
    return valor_estimado