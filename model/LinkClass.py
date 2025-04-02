from pydantic import BaseModel
from typing import Any


class LinkRaw(BaseModel):
    source: int
    target: int
    distance: float
    label: str
    
class Link(BaseModel):
    source_pos: tuple[float, float]
    target_pos: tuple[float, float]
    distance_km: float
    label: str

class LinkCrossRef(BaseModel):
    a: Any
    b: Any
    distance_km: float
    label: str

    def get_pos_a(self):
        return self.a.pos

    def get_pos_b(self):
        return self.b.pos