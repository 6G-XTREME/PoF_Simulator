from pydantic import BaseModel
from typing import List
class Node(BaseModel):
    name: str
    pos: tuple[float, float]
    node_degree: int
    type: str
    
    traffic_profile: str | None = None
    estimated_traffic_injection: float | None = None

    assoc_links: list = []
    assoc_nodes: list = []

    def add_link(self, link):
        self.assoc_links.append(link)

    def add_node_association(self, node):
        self.assoc_nodes.append(node)
        
    def to_x_y(self):
        return self.pos[0], self.pos[1]
    
    
    @staticmethod
    def obtain_x_y_vectors(nodes: list['Node']) -> tuple[list[float], list[float]]:
        x = [v.pos[0] for v in nodes]
        y = [v.pos[1] for v in nodes]
        return x, y
    
    @staticmethod
    def obtain_degree_vector(nodes: list['Node']) -> list[int]:
        return [v.node_degree for v in nodes]
    
    @staticmethod
    def obtain_name_vector(nodes: list['Node']) -> list[str]:
        return [v.name for v in nodes]
    
    @staticmethod
    def obtain_type_vector(nodes: list['Node']) -> list[str]:
        return [v.type for v in nodes]
    

    
class NodeHostingCell(Node):
    has_macrocell: bool = False
    has_femtocell: bool = False
    has_hpl: bool = False
    if_hpl__feeding_femtocells: List[Node] | None = None
    
    
