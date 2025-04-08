from pydantic import BaseModel
from model.NodeClass import Node

class Link(BaseModel):
    a: Node
    b: Node
    distance_km: float
    label: str
    name: str

    def get_pos_a(self):
        return self.a.pos

    def get_pos_b(self):
        return self.b.pos