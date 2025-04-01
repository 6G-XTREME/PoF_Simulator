from pydantic import BaseModel

class Node(BaseModel):
    id: int
    type: str
    x: float
    y: float
    node_degree: int
    name: str
