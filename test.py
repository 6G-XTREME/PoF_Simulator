import numpy as np
from matplotlib import path


def main ():
    a = np.array([[0 ,224.9842991], [0, 1000], [1000, 1000], [1000, 201.08568928]])
    _polygon =  path.Path([(a[i][0], a[i][1]) for i in range(a.shape[0])])
    print(a)
    print(_polygon.contains_points([(662.6407796509268, 71.96295428363875)]))

    np.inters
    
    print('TBD!')

if __name__ == '__main__':
    main()