"""
IR Classes for Simple SSA-like Intermediate Representation
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Union


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


@dataclass
class Op(Instruction):
    uses: List[str]
    defs: List[str]

    def __init__(self, uses: List[str] = [], defs: List[str] = []):
        super().__init__(kind="op")
        self.uses = uses
        self.defs = defs


@dataclass
class Jump(Instruction):
    targets: List[str]

    def __init__(self, targets: List[str]):
        super().__init__(kind="jmp")
        self.targets = targets


@dataclass
class PhiIncoming:
    block: str
    value: str


@dataclass
class Phi(Instruction):
    dest: str
    incomings: List[PhiIncoming]

    def __init__(self, dest: str, incomings: List[PhiIncoming]):
        super().__init__(kind="phi")
        self.dest = dest
        self.incomings = incomings

@dataclass
class Block:
    name: str
    instructions: List[Instruction]
    successors: List[str]
    predecessors: List[str]
    live_in: Union[Set[str], Dict[str, float]]
    live_out: Union[Set[str], Dict[str, float]]
    use_set: Set[str]
    def_set: Set[str]
    phi_uses: Set[str]
    phi_defs: Set[str]
    max_register_pressure: int = 0
    next_use_distances_by_val: Dict[int, List[float]] = None

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
        self.next_use_distances_by_val = {}


@dataclass
class Function:
    name: str
    blocks: Dict[str, Block]
    next_use_distances: Dict[Tuple[str, int], List[float]] = None
    value_indices: Dict[str, int] = None

    def __init__(self, name: str):
        self.name = name
        self.blocks = {}
        self.next_use_distances = {}
        self.value_indices = {}

    def add_block(self, block: Block) -> None:
        self.blocks[block.name] = block




def val_is_phi(function: Function, value: str) -> bool:
    """Returns True if the value is defined by a phi instruction."""
    for block in function.blocks.values():
        if value in block.phi_defs:
            return True
    return False


def val_as_phi(function: Function, value: str) -> Union[Phi, None]:
    """Returns the Phi instruction that defines the given value, or None if not found."""
    for block in function.blocks.values():
        if value in block.phi_defs:
            # Find the phi instruction that defines this value
            for instr in block.instructions:
                if isinstance(instr, Phi) and instr.dest == value:
                    return instr
    return None


