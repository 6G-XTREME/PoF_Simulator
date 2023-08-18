# Power over Fiber (PoF) Polling Simulator

Este repositorio contiene el código asociado para la simulación de un contexto de red telealimentado con PoF Polling, usando algoritmos de tipo DRCA (Dynamic Resource and Cell Allocation) para el provisionamiento y asignación de los recursos disponibles. 

## Características
* Implementado dos entornos de simulación:
    1. Entorno Legacy, que corresponde con la adaptación del código de MATLAB de la versión preliminar de PoF Polling.
    2. Entorno Evolved, que corresponde con una mejora del entorno legacy. Centrado principalmente en la modularidad del código para poder ejecutar correctamente tantos los algoritmos basados en el entorno legacy, como aquellos algoritmos que quieran usar el entorno Evolved.
* Implementada una interfaz gráfica que permite la ejecución y el análisis de los resultados de la simulación.
* Permite guardar los datos de simulación (CSV, gráficos y variables para replicabilidad)
* Permite ejecutar simulaciones de manera "headless" y autónoma.
* Permite la ejecución de simulaciones por "batchs", de modo que la salida de la simulación será la media de los batchs realizados.

## Ejecución del simulador
### Requerimientos
* Tener instalado Python 3.X (minimo versión 3.7)
* Instalar los modulos necesarios para ejecutar el simulador. Usar este comando en el directorio del proyecto: ``` pip install -r requirements.txt```

### Modo GUI
1. Ir al archivo: ``start_x.py``
2. Ejecutar el archivo (```python start_x.py```). Automáticamente aparecerá una interfaz gráfica para configurar y ejecutar la simulación.

*Nota: la ejecución desde la GUI solo ejecuta una iteración de simulación, no dispone de ejecución en batch ni paramétrica*
### Modo consola
1. Ir al archivo: ``run_simulator.py``
2. Configurar simulador y sus parámetros de entrada
3. El simulador se ejecutará utilizando la función: `execute_simulator`

Otras maneras de ejecutar el simulador:
* Archivo: `run_batch.py`. Objetivo: ejecutar una simulación con batch
* Archivo: `run_parametric_with_batch.py`. Objetivo: ejecutar una simulación paramétrica con simulaciones batch.

## Algoritmo DRCA E-Lighthouse

### Resumen de los cambios realizados respecto algoritmo DRCA Legacy
Partiendo del algoritmo propuesto en MATLAB, se han realizado una serie de modificaciones/extensiones que permiten mejorar el funcionamiento final del simulador, haciendolo mucho más cercano al funcionamiento real del entorno a simular. Las características añadidas son:

* Añadido concepto de tráfico. Ahora se almacenan los tráficos de los diferentes usuarios para cada instante temporal.

* Las femtocells no se encienden de manera automatica. Se ha creado un parámetro que determina cuandos slots de simulación tardan las celdas en encenderse y estar disponibles para atender a usuarios.

* Las femtocells que no están en uso por ningún usuario se apagarán de manera automatica tras un determinado tiempo sin usuarios. Si quisieramos usar esta celda de nuevo, tendremos que encenderla.

* El sistema desconoce donde están los usuarios en cada momento. Los usuarios solo reportan su ubicación cada X tiempo determinado. El sistema utilizará la última ubicación conocida para par servicio al usuario, y tomar decisiones respecto a las femtoceldas.

* Añadidas nuevas métricas para comprobar el funcionamiento de la simulación:

    - Tráfico de cada usuario por cada instante temporal.

    - Porcentaje de tiempo de simulación que un usuario esta dentro del area de una femto celda (siendo servido por ella o no).

    - Porcentaje de tiempo de simulación que un usuario esta dentro del area de una femto y esta siendo servido por ella.

    - Porcentaje de tiempo dentro del area de una femtocelda y que el usuario esta siendo servido por ella.

    - Instante temporal en el que se agota la primera batería.

    - Instante temporal en el que se agota la última batería.