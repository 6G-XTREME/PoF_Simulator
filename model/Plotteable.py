from pydantic import BaseModel

class Vertex(BaseModel):
    pos: tuple[float, float]
    degree: int
    name: str
    type: str
    
    @staticmethod
    def obtain_x_y_vectors(vertices: list['Vertex']) -> tuple[list[float], list[float]]:
        x = [v.pos[0] for v in vertices]
        y = [v.pos[1] for v in vertices]
        return x, y
    
    @staticmethod
    def obtain_degree_vector(vertices: list['Vertex']) -> list[int]:
        return [v.degree for v in vertices]
    
    @staticmethod
    def obtain_name_vector(vertices: list['Vertex']) -> list[str]:
        return [v.name for v in vertices]
    
    @staticmethod
    def obtain_type_vector(vertices: list['Vertex']) -> list[str]:
        return [v.type for v in vertices]
    
class Edge(BaseModel):
    a: tuple[float, float]
    b: tuple[float, float]
    distance: float
    label: str