from pydantic import BaseModel
from model.NodeClass import Node
import uuid
class Link(BaseModel):
    """
    Link is a class that represents a link in the network.
    """
    id: str = uuid.uuid4().hex
    a: Node
    b: Node
    distance_km: float
    label: str
    name: str

    def get_pos_a(self):
        return self.a.pos

    def get_pos_b(self):
        return self.b.pos