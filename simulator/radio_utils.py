__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro"]
__version__ = "1.0.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Working on"

import numpy as np

def compute_sinr_dl(P, BaseStations, closest, alpha, Pm, Pf, NMacro, noise):
    """ 
        Compute SINR for Downlink
    """
    
    # Convert P to Numpy
    P = np.array(P)
    closest = int(closest)
    
    # Compute signal power.
    distance = np.linalg.norm(np.subtract(P, BaseStations[closest][0:2]))
    if closest < NMacro:
        Power = Pm
        # freq=1800e9;
    else:
        Power = Pf
        # freq = 5e9;
    
    hx = 1  # raylrnd(b)
    Signal = 10 * np.log10(Power * hx * distance ** (-alpha))
    # Signal = 10*log10(Power*hx*((3e8/freq)/4*pi*distance)^alpha)

    s = BaseStations.shape
    Interferers = np.setdiff1d(np.arange(0, s[0]), closest)

    FinalInterference = 0

    # Compute Interference.
    for k in Interferers:
        if k < NMacro:
            Int_Power = Pm
        else:
            Int_Power = Pf

        h = 1  # raylrnd(b)
        dist = np.linalg.norm(np.subtract(P, BaseStations[k][0:2]))

        Interference = Int_Power * h * dist ** (-alpha)
        # Interference = 10*log10(Int_Power*h*((3e8/freq)/4*pi*dist)^alpha)

        # Add the contribution.
        FinalInterference = FinalInterference + Interference

    # Compute SINR adding the noise.
    FinalInterference = FinalInterference + noise
    sinr = Signal - (10 * np.log10(FinalInterference))
    
    return sinr

def compute_sinr_ul ():
    print("TBD!")