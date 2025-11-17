"""
IR Classes for Simple SSA-like Intermediate Representation
"""

from dataclasses import dataclass
from typing import List, Dict, Set


@dataclass
class ParseError(Exception):
    message: str
    line: int
    column: int = 0

    def __str__(self) -> str:
        return f"ParseError at line {self.line}, column {self.column}: {self.message}"


@dataclass
class Instruction:
    kind: str  # "op", "jmp", or "phi"
    val_local_idx: int


@dataclass
class Op(Instruction):
    uses: List[str]
    defs: List[str]

    def __init__(self, val_local_idx: int, uses: List[str] = [], defs: List[str] = []):
        super().__init__(kind="op", val_local_idx=val_local_idx)
        self.uses = uses
        self.defs = defs


@dataclass
class Jump(Instruction):
    targets: List[str]

    def __init__(self, val_local_idx: int, targets: List[str]):
        super().__init__(kind="jmp", val_local_idx=val_local_idx)
        self.targets = targets


@dataclass
class PhiIncoming:
    block: str
    value: str


@dataclass
class Phi(Instruction):
    dest: str
    incomings: List[PhiIncoming]

    def __init__(self, val_local_idx: int, dest: str, incomings: List[PhiIncoming]):
        super().__init__(kind="phi", val_local_idx=val_local_idx)
        self.dest = dest
        self.incomings = incomings


@dataclass
class Block:
    name: str
    instructions: List[Instruction]
    successors: List[str]
    predecessors: List[str]
    live_in: Set[str]
    live_out: Set[str]
    use_set: Set[str]
    def_set: Set[str]
    phi_uses: Set[str]
    phi_defs: Set[str]

    def __init__(self, name: str):
        self.name = name
        self.instructions = []
        self.successors = []
        self.predecessors = []
        self.live_in = set()
        self.live_out = set()
        self.use_set = set()
        self.def_set = set()
        self.phi_uses = set()
        self.phi_defs = set()


@dataclass
class Function:
    name: str
    blocks: Dict[str, Block]

    def __init__(self, name: str):
        self.name = name
        self.blocks = {}

    def add_block(self, block: Block) -> None:
        self.blocks[block.name] = block
