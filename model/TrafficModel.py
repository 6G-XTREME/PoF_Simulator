import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Tiempo en intervalos de 15 minutos
tiempo = np.arange(0, 24, 0.25)  # 96 puntos


# Función modificada con crecimiento desde 10am y pico en 8pm
def valley_peak_traffic_profile(t_segundos):
    """
    Devuelve el perfil de tráfico evaluado para tiempo en segundos.
    """
    t_horas = t_segundos / 3600.0

    # Componente de subida desde 10am
    subida = 1 / (1 + np.exp(-(t_horas - 10)))

    # Componente pico gaussiano a las 20:00
    pico = np.exp(-0.5 * ((t_horas - 20) / 2) ** 2)

    # Combinación y normalización
    combinado = 0.6 * subida + 0.4 * pico
    normalizado = combinado / np.max(combinado)

    return 0.1 + 0.9 * normalizado



def estimate_traffic_from_seconds(t_segundos, ruido=True, ruido_max=0.015, seed=None):
    """
    Devuelve el factor de tráfico para un instante temporal dado en segundos.
    Usa interpolación con resolución de 15 minutos (900s) y ruido opcional.
    """
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

    return np.clip(valor_estimado, 0.1, 1.0)

