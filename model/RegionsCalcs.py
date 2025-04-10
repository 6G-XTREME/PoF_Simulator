__author__ = "Antonio Ginés Buendía López (abuendia@e-lighthouse.com)"
__credits__ = ["Antonio Ginés Buendía López", "Gabriel Otero Perez", "Jose-Manuel Martinez-Caro"]
__version__ = "1.0.0"
__maintainer__ = "Antonio Ginés Buendía López"
__email__ = "abuendia@e-lighthouse.com"
__status__ = "Development"

import numpy as np
import sys
from math import pi, sqrt
from simulator.bcolors import bcolors
from simulator.polygon_cut import polyclip
from shapely.geometry import Point, GeometryCollection, MultiPolygon, Polygon



# -------------------------------------------------------------------------------------------------------------------- #
# -- Create regions -------------------------------------------------------------------------------------------------- #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
def create_regions(
        BaseStations,
        alpha_loss,
        polygon_bounds: list[tuple[float, float]] = [(0,0), (0,1000), (1000,1000),(1000, 0), (0,0)],
        euclidean_to_m_factor: float = 1,
        use_power_based_radius: bool = False,
        max_radius_m_list: list[float] = None,
    ):
    """
    Create regions for the coverage of the base stations.
    
    Args:
        BaseStations: list - List of base stations. Each base station is a tuple (x (m), y (m), p_tx (W)). 
        alpha_loss: float - Alpha loss
        polygon_bounds: list[tuple[float, float]] - Bounds of the polygon that represents the whole region in meters
        euclidean_to_m_factor: float - Factor to convert euclidean distance to meters
        use_power_based_radius: bool - Whether to use power-based radius calculation
        max_radius_m_list: list[float] - List of maximum radius in meters for each base station
    """
    _WholeRegion = Polygon(polygon_bounds)
    if not _WholeRegion.is_valid:
        _WholeRegion = _WholeRegion.buffer(0)
    _UnsoldRegion = _WholeRegion
    Regions = {}
    
    # Convert coordinates to meters if needed
    BaseStations_m = np.array(BaseStations)
    if euclidean_to_m_factor != 1:
        BaseStations_m[:, :2] *= euclidean_to_m_factor
        max_radius_m_list[:] *= euclidean_to_m_factor
    
    Npoints = len(BaseStations)
    for k in range(Npoints-1,-1,-1):
        _Region = _UnsoldRegion
        
        # Calculate maximum radius for this base station
        if use_power_based_radius:
            # Convert power to dBm (assuming input is in W)
            p_tx_dbm = 10 * np.log10(BaseStations[k, 2] * 1000)  # Convert W to mW to dBm
            max_radius_m = calculate_cell_radius(p_tx_dbm)
        elif max_radius_m_list is not None:
            max_radius_m = max_radius_m_list[k]
        else:
            max_radius_m = 1000  # Default 1km radius
            
        # Create circular region for maximum coverage
        max_coverage = Point(BaseStations_m[k, 0], BaseStations_m[k, 1]).buffer(max_radius_m)
        _Region = _Region.intersection(max_coverage)
        
        for j in range(0, Npoints):
            if j < k:
                if BaseStations[k, 2] != BaseStations[j, 2]:
                    _resp = apollonius_circle_path_loss(
                        BaseStations_m[k][:2], 
                        BaseStations_m[j][:2], 
                        BaseStations[k][2], 
                        BaseStations[j][2], 
                        alpha_loss
                    )
                    _Circ = get_circle(_resp)
                    _Reg2 = Polygon(_Circ)
                    if not _Reg2.is_valid:
                        _Reg2 = _Reg2.buffer(0)
                    _Region = _Region.buffer(0.0001).intersection(_Reg2.buffer(0.0001))
                else:
                    _R = get_dominance_area(BaseStations_m[k][:2], BaseStations_m[j][:2], polygon_bounds)
                    if not _R.is_valid:
                        _R = _R.buffer(0)
                    _Region = _Region.buffer(0.0001).intersection(_R.buffer(0.0001))
        
        Regions[k] = _Region
        
    return Regions










# -------------------------------------------------------------------------------------------------------------------- #
# -- Calculate cell radius -------------------------------------------------------------------------------------------- #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
def calculate_cell_radius(p_tx_dbm, sensitivity_dbm=-90, frequency_mhz=900, path_loss_exponent=2.5):
    """
    Calculate the cell radius based on path loss model.
    
    Args:
        p_tx_dbm: float - Transmitted power in dBm
        sensitivity_dbm: float - Receiver sensitivity in dBm
        frequency_mhz: float - Frequency in MHz
        path_loss_exponent: float - Path loss exponent (typically 2-4)
    
    Returns:
        float - Cell radius in meters
    """
    # Convert frequency to Hz
    frequency_hz = frequency_mhz * 1e6
    
    # Free space path loss at 1m
    fsl_1m = 20 * np.log10(4 * np.pi * 1 * frequency_hz / 3e8)
    
    # Calculate maximum path loss
    max_path_loss = p_tx_dbm - sensitivity_dbm - fsl_1m
    
    # Calculate radius using path loss model
    radius_m = 10 ** (max_path_loss / (10 * path_loss_exponent))
    
    return radius_m



# -------------------------------------------------------------------------------------------------------------------- #
# -- Calculate Apollonius circle -------------------------------------------------------------------------------------- #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
def apollonius_circle_path_loss(P1, P2, w1, w2, alpha):
    """
    Calculate the Apollonius circle for two points based on path loss model.
    This function assumes input coordinates are in meters and power values are in mW.
    
    Args:
        P1: tuple[float, float] - First point coordinates in meters
        P2: tuple[float, float] - Second point coordinates in meters
        w1: float - First point's transmitted power in mW
        w2: float - Second point's transmitted power in mW
        alpha: float - Path loss exponent
        
    Returns:
        tuple[float, float, float] - Center (x, y) in meters and radius in meters of the Apollonius circle
    """
    try:
        # Convert power to dBm for calculations
        w1_dbm = 10 * np.log10(w1 * 1000)  # Convert mW to dBm
        w2_dbm = 10 * np.log10(w2 * 1000)  # Convert mW to dBm
        
        # Calculate lambda based on power difference
        _lambda = 10 ** ((w1_dbm - w2_dbm) / (10 * alpha))
        
        # Calculate circle center and radius
        _Cx = (P1[0] - P2[0] * _lambda**2) / (1 - _lambda**2)
        _Cy = (P1[1] - P2[1] * _lambda**2) / (1 - _lambda**2)
        
        # Calculate radius in meters
        _r = _lambda * np.sqrt((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2) / abs(1 - _lambda**2)
        
        return _Cx, _Cy, _r

    except Exception as e:
        print(bcolors.FAIL + 'Error in function: ' + sys._getframe().f_code.co_name + bcolors.ENDC)
        print(bcolors.FAIL + 'Error in file: '+ sys._getframe().f_code.co_filename + bcolors.ENDC)
        print(e)



# -------------------------------------------------------------------------------------------------------------------- #
# -- Get circle ------------------------------------------------------------------------------------------------------ #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
def get_circle(var):
    """
    Get the circle for a given point and radius.
    Args:
        var: tuple[float, float, float] - Center (x, y) and radius of the circle.
    Returns:
        list[tuple[float, float]] - List of points that form the circle.
    """
    try:
        _aux = pi/50
        _th = np.arange(0,(2*pi)+_aux,_aux)

        xunit = var[2] * np.cos(_th) + var[0]
        yunit = var[2] * np.sin(_th) + var[1]

        res = [(x, y) for x, y in zip(xunit, yunit)]
        return res
    except Exception as e:
        print(bcolors.FAIL + 'Error in function: ' + sys._getframe( ).f_code.co_name + bcolors.ENDC)
        print(bcolors.FAIL + 'Error in file: '+ sys._getframe( ).f_code.co_filename + bcolors.ENDC)
        print(e)


# -------------------------------------------------------------------------------------------------------------------- #
# -- Get dominance area ---------------------------------------------------------------------------------------------- #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
def get_dominance_area(P1, P2, polygon_bounds: list[tuple[float, float]] = [(0,0), (0,1000), (1000,1000), (1000, 0)]):
    """
    Get the dominance area for a given point and another point.
    Args:
        P1: tuple[float, float] - First point coordinates in meters
        P2: tuple[float, float] - Second point coordinates in meters
    Returns:
        Polygon - Dominance area in meters.
    """
    _medZero, _medOne = perpendicular_bisector(P1, P2)
    _WholeRegion = Polygon(polygon_bounds)
    
    _c = polyclip(_WholeRegion, [0, _medZero], [1, _medOne])

    _point = Point(P1[0], P1[1])    
    _polygon = Polygon(_c)
    
    if(_polygon.contains(_point) is False):
        _Reg1 = Polygon(_WholeRegion)
        _Reg = _Reg1.difference(_polygon)
        return _Reg
    else:
        return _polygon


# -------------------------------------------------------------------------------------------------------------------- #
# -- Get Euclidean distance ------------------------------------------------------------------------------------------- #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
def get_euclidean_distance(X, Y):
    """
    Get the Euclidean distance between two points in meters.
    Args:
        X: tuple[float, float] - First point coordinates in meters
        Y: tuple[float, float] - Second point coordinates in meters
    Returns:
        float - Euclidean distance in meters.
    """
    return sqrt((X[0]-Y[0])**2 + (X[1]-Y[1])**2)


# -------------------------------------------------------------------------------------------------------------------- #
# -- Get distance in kilometers --------------------------------------------------------------------------------------- #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
def get_distance_in_kilometers(X, Y):
    """
    Get the distance in kilometers between two points.
    Args:
        X: tuple[float, float] - First point coordinates in meters
        Y: tuple[float, float] - Second point coordinates in meters
    Returns:
        float - Distance in kilometers.
    """
    # Calculate Euclidean distance in meters
    d_meters = get_euclidean_distance(X, Y)
    
    # Convert to kilometers
    return d_meters / 1000


# -------------------------------------------------------------------------------------------------------------------- #
# -- Get perpendicular bisector ---------------------------------------------------------------------------------------- #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
def perpendicular_bisector(P1, P2):
    """
    Get the perpendicular bisector for two points.
    Args:
        P1: tuple[float, float] - First point
        P2: tuple[float, float] - Second point
    Returns:
        tuple[float, float] - Slope and intercept of the perpendicular bisector.
    """
    
    _xmed = (P1[0] + P2[0])/2
    _ymed = (P1[1] + P2[1])/2

    _a = -1/((P2[1] - P1[1])/(P2[0] - P1[0]))
    _b = _ymed - (_a*_xmed)

    return _b, (_a + _b)


# -------------------------------------------------------------------------------------------------------------------- #
# -- Search for the closest base station ------------------------------------------------------------------------------ #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
def search_closest_bs(point, regions, backup_regions = None):
    """
    Search for the closest base station for a given point.
    Args:
        point: tuple[float, float] - Point
        regions: list - List of regions from femtocells.
        backup_regions: list - List of backup regions from macrocells.
    Returns:
        int - Index of the closest base station.
    """
    
    # Regions are sorted from lowest to highest preference or weight.
    closest = 0

    for region_idx in range(len(regions)):
        if isinstance(regions[region_idx], Polygon):
            polygon = regions[region_idx]
            if polygon.contains(Point(point)):
                closest = region_idx
        # Undetermined case that a region is a MultiPolygon... 
        elif isinstance(regions[region_idx], MultiPolygon):
            multipolygon = regions[region_idx]
            poly = multipolygon.envelope
            if poly.contains(Point(point)):
                closest = region_idx
        # Undetermined case that a region is a GeometryCollection...
        elif isinstance(regions[region_idx], GeometryCollection):
            poly = regions[region_idx].convex_hull
            if poly.contains(Point(point)):
                closest = region_idx
                
                
    # TODO: Add backup regions

    return closest
