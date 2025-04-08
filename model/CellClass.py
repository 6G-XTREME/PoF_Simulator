from pydantic import BaseModel
import uuid
from shapely.geometry import Polygon



class MacroCell(BaseModel):
    """
    MacroCell is a class that represents a macrocell in the network.
    """
    id: str = uuid.uuid4().hex
    pos: tuple[float, float]
    hosting_node_id: str
    capacity: float | None = None
    radius: float | None = None
    transmission_power: float | None = None
    coverage_shape: Polygon | None = None
    
    
class FemtoCell(BaseModel):
    """
    FemtoCell is a class that represents a femtocell in the network.
    """
    id: str = uuid.uuid4().hex
    pos: tuple[float, float]
    hosting_node_id: str
    capacity: float | None = None
    transmission_power: float | None = None
    coverage_shape: Polygon | None = None
    
    
    
    