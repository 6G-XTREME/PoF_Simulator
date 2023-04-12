# PoF_Simulation

This repository contains the code to simulate PoF system, implementing some algorithms of decisions in order to experiment with differents behaviour of the system.

## Features
* Execute legacy algorithm for PoF (developed with MATLAB and converted to Python)
* Save simulation data (csv, plots and experiment data)
* Execute the simulator headless
* System ready to do batch simulations

## Execute Simulator

1. Go to file: ``run_simulator.py``
2. Configure simulator and input_parameters
3. Execute the simulator with function `execute_simulator`

## E-Lighthouse Algorithm

### Resumen de los cambios realizados respecto UC3M
Partiendo del algoritmo propuesto por UC3M en MATLAB, se han realizado una serie de modificaciones/extensiones que permiten mejorar el funcionamiento final del simulador, haciendolo mucho más cercano al funcionamiento real del entorno a simular. Las características añadidas son:

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