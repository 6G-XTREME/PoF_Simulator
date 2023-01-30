__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro"]
__version__ = "1.0.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Working on"

import numpy as np

def clip (p1, p2, plane):
    _d1 = p1[0] * plane [0] + p1[1] * plane[1] - plane[2]
    _d2 = p2[0] * plane [0] + p2[1] * plane[1] - plane[2]
    _t = (0 - _d1)/(_d2 - _d1)
    return p1 + _t * (p2 - p1)
    print("TBD!")

def inside (p, plane):
    _d = p[0] * plane[0] + p[1] * plane[1]
    if (_d > plane[2]): return True;
    else: return False;

def polyclip (pin, x1, x2):
    _plane = [x1[1] - x2[1], x2[0] - x1[0], 0]
    _plane[2] = x1[0] * _plane[0] + x1[1] * _plane[1]
    
    _n = pin.shape[0]
    _s = pin[-1, :]
    _pout = np.empty((0,2))

    for ci in range(_n):
        _p = pin[ci, :]
        if (inside(_p, _plane)):
            if (inside(_s, _plane)): _pout = np.append(_pout, np.array([_p]),axis=0)
            else: 
                _t = clip(_s,_p,_plane)
                _pout = np.append(_pout, np.array([_t]),axis=0)
                _pout = np.append(_pout, np.array([_p]),axis=0)
        else:
            if(inside(_s, _plane)): #case 2
                _t=clip(_s, _p, _plane)
                _pout = np.append(_pout, np.array([_t]),axis=0)
        _s = _p
    return np.array(_pout)

def test ():
    print("TBD!")