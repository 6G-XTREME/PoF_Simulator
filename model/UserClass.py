from pydantic import BaseModel


# TODO: not definitive, just ideas
class User(BaseModel):
    x: float
    y: float
    name: str