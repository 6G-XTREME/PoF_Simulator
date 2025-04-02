from pydantic import BaseModel
from model.NodeClass import Node
from model.LinkClass import LinkRaw

class FileFormat(BaseModel):
    nodes: list[Node]
    links: list[LinkRaw]