from pydantic import BaseModel

class Node(BaseModel):
    id: int
    type: str
    x: float
    y: float
    node_degree: int
    name: str

class NodeCrossRef(BaseModel):
    name: str
    pos: tuple[float, float]
    node_degree: int
    type: str

    assoc_links: list = []
    assoc_nodes: list = []

    def add_link(self, link):
        self.assoc_links.append(link)

    def add_node_association(self, node):
        self.assoc_nodes.append(node)
