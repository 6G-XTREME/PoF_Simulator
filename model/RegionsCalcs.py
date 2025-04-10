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




# ---------------------------------------------------------------------------------------------------------------- #
# -- Create regions for the coverage of the base stations -------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------- #
def create_regions(
        BaseStations,
        alpha_loss,
        polygon_bounds: list[tuple[float, float]] = [(0,0), (0,1000), (1000,1000),(1000, 0), (0,0)],
        euclidean_to_km_scale: float = 1,
        use_power_based_radius: bool = False,
        max_radius_km_list: list[float] = None,
    ):
    """
    Create regions for the coverage of the base stations.
    
    Args:
        BaseStations: list - List of base stations. Each base station is a tuple (x, y, p_tx). x,y in km, p_tx in W.
        alpha_loss: float - Path loss exponent.
        polygon_bounds: list[tuple[float, float]] - Bounds of the polygon that represents the whole region.
        euclidean_to_km_scale: float - Scale factor to convert map units to kilometers.
        use_power_based_radius: bool - Whether to use power-based radius calculation.
        max_radius_km_list: list[float] - List of maximum radii in kilometers for each base station.
    """
    # Create the whole region with a small buffer to ensure validity
    _WholeRegion = Polygon(polygon_bounds)
    if not _WholeRegion.is_valid:
        _WholeRegion = _WholeRegion.buffer(0)
    _UnsoldRegion = _WholeRegion
    Regions = {}
    Npoints = len(BaseStations)
    default_coverage_radius_km = 1
    
    
    # Calculate maximum coverage radius for all base stations
    radius_km = []
    for i in range(Npoints):
        max_radius = max_radius_km_list[i] if max_radius_km_list is not None else default_coverage_radius_km
        radius = calculate_radius_km(BaseStations[i][2], alpha_loss, euclidean_to_km_scale)
        radius_km.append(radius)
    
    # Resolve conflicts and create regions
    BaseStations = np.array(BaseStations)
    for k in range(Npoints-1, -1, -1):
        _Region = _UnsoldRegion     # Initialize the region with the unsold region
            
        # Create circular region for maximum coverage
        max_coverage = Point(BaseStations[k,0], BaseStations[k,1]).buffer(radius_km[k])
        _Region = _Region.intersection(max_coverage)
        
        # # Start with the base station's coverage circle
        # _Region = _Region.intersection(max_coverage)
        # if not _Region.is_valid:
        #     _Region = _Region.buffer(0)
        
        # Handle conflicts with other base stations
        for j in range(Npoints):
            if j < k:
                
                this_bs = BaseStations[k]
                other_bs = BaseStations[j]
                
                # If base stations have different powers, use Apollonius circle
                if this_bs[2] != other_bs[2]:
                    _resp = apollonius_circle_path_loss(
                        this_bs[0:2],
                        other_bs[0:2],
                        this_bs[2],
                        other_bs[2],
                        alpha_loss
                    )
                    # Get the circle from the apollonius circle
                    _Circ = get_circle(_resp)
                    
                    # Create a buffer around the circle
                    _Reg2 = Polygon(_Circ).buffer(0.0001)
                    if not _Reg2.is_valid:
                        _Reg2 = _Reg2.buffer(0)
                    
                    # Intersect the circle with the maximum coverage of the bs
                    _Reg2 = _Reg2.intersection(max_coverage)
                    if not _Reg2.is_valid:
                        _Reg2 = _Reg2.buffer(0)

                    # Remove the circle from the global region
                    _Region = _Region.intersection(_Reg2)
                    if not _Region.is_valid:
                        _Region = _Region.buffer(0)
                        
                else:
                    # If same power, use dominance area
                    _R = get_dominance_area(this_bs[0:2], other_bs[0:2])
                    
                    # Create a buffer around the dominance area
                    _R = _R.buffer(0.0001)
                    if not _R.is_valid:
                        _R = _R.buffer(0)
                    
                    # Intersect the dominance area with the maximum coverage of the bs
                    _R = _R.intersection(max_coverage)
                    if not _R.is_valid:
                        _R = _R.buffer(0)
                    
                    # 
                    _Region = _Region.intersection(_R)
                    if not _Region.is_valid:
                        _Region = _Region.buffer(0)
        
        # Ensure the final region is valid
        if not _Region.is_valid:
            _Region = _Region.buffer(0)
        
        if not _UnsoldRegion.is_valid:
            _UnsoldRegion = _UnsoldRegion.buffer(0)
        
        Regions[k] = _Region
        _UnsoldRegion = _UnsoldRegion.difference(_Region)
        if not _UnsoldRegion.is_valid:
            _UnsoldRegion = _UnsoldRegion.buffer(0)
    
    return Regions




def create_base_region_for_bs(
        bs_position: tuple[float, float],
        bs_power: float,
        alpha_loss: float,
        max_radius_km: float = 1,
        euclidean_to_km_scale: float = 1
    ):
    """
    Create a base region for a base station.
    """
    # Calculate the radius of coverage for the base station
    radius_km = calculate_radius_km(bs_power, alpha_loss, euclidean_to_km_scale)
    
    # Create a circular region for the base station
    region = Point(bs_position[0], bs_position[1]).buffer(radius_km)
    
    return region


# ---------------------------------------------------------------------------------------------------------------- #
# -- Calculate radius --------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------- #
def calculate_radius_km(power: float, alpha_loss: float, euclidean_to_km_scale: float = 1) -> float:
    """
    Calculate the radius of coverage for a base station based on its power or max radius.
    
    Args:
        power: float - Transmit power of the base station
        alpha_loss: float - Path loss exponent
        euclidean_to_km_scale: float - Scale factor to convert map units to kilometers
        
    Returns:
        float - Radius in map units
    """
    
    # Otherwise calculate based on power
    # Using a simplified path loss model: P_r = P_t * d^(-alpha)
    # We'll use a minimum received power threshold of -100 dBm (1e-10 mW)
    min_rx_power = 1e-10  # -100 dBm
    radius_eu = (power / min_rx_power) ** (1/alpha_loss)
    radius_m = radius_eu * euclidean_to_km_scale
    return radius_m / 1000




# ---------------------------------------------------------------------------------------------------------------- #
# -- Apollonius circle path loss ---------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------- #
def apollonius_circle_path_loss (P1, P2, w1, w2, alpha):
    """
    Apollonius circle path loss is a function that calculates the apollonius circle for two points and two weights.
    For a radio signal, the interpretation of the function 
    Args:
        P1: tuple[float, float] - First point
        P2: tuple[float, float] - Second point
        w1: float - First weight
        w2: float - Second weight
        alpha: float - Alpha
    Returns:
        tuple[float, float, float] - Center (x, y) and radius of the apollonius circle
    """
    try:
        _lambda = (w1/w2)**(1/alpha)
        _lambda = _lambda.real

        _Cx = (P1[0] - P2[0]*_lambda**2)/(1-_lambda**2)
        _Cy = (P1[1] - P2[1]*_lambda**2)/(1-_lambda**2)

        _r = _lambda * sqrt((P1[0] - P2[0])**2 + (P1[1] -P2[1])**2) /  np.linalg.norm(1 - _lambda**2)

        return _Cx, _Cy, _r

    except Exception as e:
        print(bcolors.FAIL + 'Error in function: ' + sys._getframe( ).f_code.co_name + bcolors.ENDC)
        print(bcolors.FAIL + 'Error in file: '+ sys._getframe( ).f_code.co_filename + bcolors.ENDC)
        print(e)



# ---------------------------------------------------------------------------------------------------------------- #
# -- Get circle -------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------- #
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



# ---------------------------------------------------------------------------------------------------------------- #
# -- Get dominance area -------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------- #
def get_dominance_area(P1, P2, polygon_bounds: list[tuple[float, float]] = [(0,0), (0,1000), (1000,1000), (1000, 0)]):
    """
    Get the dominance area for a given point and another point.
    Args:
        P1: tuple[float, float] - First point
        P2: tuple[float, float] - Second point
    Returns:
        Polygon - Dominance area.
    """
    _medZero, _medOne = perpendicular_bisector(P1, P2)
    _WholeRegion = Polygon(polygon_bounds)
    
    _c =polyclip(_WholeRegion, [0, _medZero], [1, _medOne])

    _point = Point(P1[0], P1[1])    
    _polygon = Polygon(_c)
    
    
    if(_polygon.contains(_point) is False):
        _Reg1 = Polygon(_WholeRegion)
        
        _Reg = _Reg1.difference(_polygon)
        # xx, yy = _Reg.exterior.coords.xy
        # _a = xx.tolist()                    
        # _b = yy.tolist()
        return _Reg
    else:
        # _a = _c[:,0]
        # _b = _c[:,1]
        return _polygon




# ---------------------------------------------------------------------------------------------------------------- #
# -- Get Euclidean distance ---------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------- #
def get_euclidean_distance(X, Y):
    """
    Get the Euclidean distance between two points.
    Args:
        X: tuple[float, float] - First point
        Y: tuple[float, float] - Second point
    Returns:
        float - Euclidean distance.
    """
    return sqrt((X[0]-Y[0])**2 + (X[1]-Y[1])**2)



# ---------------------------------------------------------------------------------------------------------------- #
# -- Get distance in kilometers ------------------------------------------------------------------------------------ #
# ---------------------------------------------------------------------------------------------------------------- #
def get_distance_in_kilometers(X, Y, euclidean_to_km_scale: float = 1):
    """
    Get the distance in kilometers between two points.
    Args:
        X: tuple[float, float] - First point
        Y: tuple[float, float] - Second point
        euclidean_to_km_scale: float - Scale of the map.
    Returns:
        float - Distance in kilometers.
    """
    
    # Calculate Euclidean distance in map units
    d_units = get_euclidean_distance(X, Y)

    # Convert distance to kilometers, based on the provided scale
    return d_units * euclidean_to_km_scale



# ---------------------------------------------------------------------------------------------------------------- #
# -- Get perpendicular bisector ------------------------------------------------------------------------------------ #
# ---------------------------------------------------------------------------------------------------------------- #
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



# ---------------------------------------------------------------------------------------------------------------- #
# -- Search for the closest base station -------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------- #
def search_closest_bs(P, Regions):
    """
    Search for the closest base station for a given point.
    Args:
        P: tuple[float, float] - Point
        Regions: list - List of regions.
    Returns:
        int - Index of the closest base station.
    """
    
    # Regions are sorted from lowest to highest preference or weight.
    closest = 0

    for region_idx in range(len(Regions)):
        if isinstance(Regions[region_idx], Polygon):
            polygon = Regions[region_idx]
            if polygon.contains(Point(P)):
                closest = region_idx
        # Undetermined case that a region is a MultiPolygon... 
        elif isinstance(Regions[region_idx], MultiPolygon):
            multipolygon = Regions[region_idx]
            poly = multipolygon.envelope
            if poly.contains(Point(P)):
                closest = region_idx
        # Undetermined case that a region is a GeometryCollection...
        elif isinstance(Regions[region_idx], GeometryCollection):
            poly = Regions[region_idx].convex_hull
            if poly.contains(Point(P)):
                closest = region_idx

    return closest
