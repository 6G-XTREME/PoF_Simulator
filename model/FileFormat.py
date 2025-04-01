from pydantic import BaseModel
from model.NodeClass import Node
from model.LinkClass import Link

class FileFormat(BaseModel):
    nodes: list[Node]
    links: list[Link]