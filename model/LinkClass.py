from pydantic import BaseModel
class Link(BaseModel):
    source: int
    target: int
    distance: float
    color: str
    label: str