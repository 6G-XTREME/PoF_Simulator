__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro"]
__version__ = "1.0.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Working on"

from math import pi
import numpy as np

def apollonius_circle_path_loss ():
    print("TBD!")

def get_circle(x, y, r):
    aux = pi/50
    th = np.arange(0,(2*pi)+aux,aux)

    xunit = r * np.cos(th) + x
    yunit = r * np.sin(th) + y

    return xunit, yunit

def get_dominance_area():
    print("TBD!")

def get_euclidean_distance():
    print("TBD!")

def perpendicular_bisector():
    print("TBD!")

def search_closest_bs():
    print("TBD!")

