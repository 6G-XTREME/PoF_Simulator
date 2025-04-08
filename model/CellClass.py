from pydantic import BaseModel
import uuid

class MacroCell(BaseModel):
    """
    MacroCell is a class that represents a macrocell in the network.
    """
    id: str = uuid.uuid4().hex
    pos: tuple[float, float]
    hosting_node_id: str
    capacity: float | None = None
    radius: float | None = None
    
    
class FemtoCell(BaseModel):
    """
    FemtoCell is a class that represents a femtocell in the network.
    """
    id: str = uuid.uuid4().hex
    pos: tuple[float, float]
    hosting_node_id: str
    capacity: float | None = None
    radius: float | None = None