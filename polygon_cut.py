__author__ = "Gabriel Otero Perez (gaoterop@it.uc3m.es), Jose-Manuel Martinez-Caro (jmmartinez@e-lighthouse.com)"
__credits__ = ["Gabriel Otero Perez", "Jose-Manuel Martinez-Caro"]
__version__ = "1.0.1"
__maintainer__ = "Jose-Manuel Martinez-Caro"
__email__ = "jmmartinez@e-lighthouse.com"
__status__ = "Working on"

import numpy as np
import operator

def clip (p1, p2, plane):
    _d1 = p1[0] * plane [0] + p1[1] * plane[1] - plane[2]
    _d2 = p2[0] * plane [0] + p2[1] * plane[1] - plane[2]
    _t = (0 - _d1)/(_d2 - _d1)
    return tuple(map(operator.add, p1, map(lambda x: _t * x, map(operator.sub, p2, p1))))
    print("TBD!")

def inside (p, plane):
    _d = p[0] * plane[0] + p[1] * plane[1]
    if (_d > plane[2]): return True;
    else: return False;

def polyclip (pin, x1, x2):
    _plane = [x1[1] - x2[1], x2[0] - x1[0], 0]
    _plane[2] = x1[0] * _plane[0] + x1[1] * _plane[1]
    
    _n = len(list(pin.exterior.coords))
    _s = list(pin.exterior.coords)[-1]
    if pin.exterior.coords[0] == pin.exterior.coords[-1]: 
        _n -= 1
        _s = list(pin.exterior.coords)[-2]
    
    _pout = []

    for ci in range(_n):
        _p = list(pin.exterior.coords)[ci]
        if (inside(_p, _plane)):
            if (inside(_s, _plane)): _pout.append(_p)
            else: 
                _t = clip(_s,_p,_plane)
                _pout.append(_t)
                _pout.append(_p)
        else:
            if(inside(_s, _plane)): #case 2
                _t=clip(_s, _p, _plane)
                _pout.append(_t)
        _s = _p
    return _pout
