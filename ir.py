"""
IR Classes for Simple SSA-like Intermediate Representation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Union, Optional

# Use u32-style maximum to represent "infinity" for next-use distances.
U32_MAX: int = (1 << 32) - 1


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
    use_colors: Dict[str, int] = field(default_factory=dict)
    def_colors: Dict[str, int] = field(default_factory=dict)

    def __init__(self, uses: List[str] = [], defs: List[str] = []):
        super().__init__(kind="op")
        self.uses = uses
        self.defs = defs
        self.use_colors = {}
        self.def_colors = {}


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
    dest_color: Optional[int] = None

    def __init__(self, dest: str, incomings: List[PhiIncoming]):
        super().__init__(kind="phi")
        self.dest = dest
        self.incomings = incomings
        self.dest_color = None

    def incoming_val_for_block(self, block_name: str) -> Union[str, None]:
        """Return the incoming value from a given predecessor block, or None if not found."""
        for incoming in self.incomings:
            if incoming.block == block_name:
                return incoming.value
        return None

    def incoming_block_for_slot(self, slot_idx: int) -> Union[str, None]:
        """Return the predecessor block name at a given slot index, or None if out of bounds."""
        if 0 <= slot_idx < len(self.incomings):
            return self.incomings[slot_idx].block
        return None

@dataclass
class Block:
    name: str
    instructions: List[Instruction]
    successors: List[str]
    predecessors: List[str]
    live_in: Dict[int, int]
    live_out: Dict[int, int]
    use_set: Set[int]
    def_set: Set[int]
    phi_uses: Set[int]
    phi_defs: Set[int]
    max_register_pressure: int = 0
    next_use_distances_by_val: Dict[int, List[int]] = None
    loop_depth: int = 0

    def __init__(self, name: str):
        self.name = name
        self.instructions = []
        self.successors = []
        self.predecessors = []
        self.live_in = {}
        self.live_out = {}
        self.use_set = set()
        self.def_set = set()
        self.phi_uses = set()
        self.phi_defs = set()
        self.next_use_distances_by_val = {}
        self.loop_depth = 0

    def phis(self):
        """Iterate over phi instructions at the start of the block."""
        for instr in self.instructions:
            if isinstance(instr, Phi):
                yield instr
            else:
                break  # Phis are always at the beginning


@dataclass
class Function:
    name: str
    blocks: Dict[str, Block]
    next_use_distances: Dict[Tuple[str, int], List[int]] = None
    value_indices: Dict[str, int] = None

    def __init__(self, name: str):
        self.name = name
        self.blocks = {}
        self.next_use_distances = {}
        self.value_indices = {}

    def add_block(self, block: Block) -> None:
        self.blocks[block.name] = block




def get_val_name(function: Function, val_idx: int) -> str:
    """Get variable name from val_idx."""
    for var, idx in function.value_indices.items():
        if idx == val_idx:
            return var
    raise ValueError(f"val_idx {val_idx} not found in value_indices")


def val_is_phi(function: Function, val_idx: int) -> bool:
    """Returns True if the value (by val_idx) is defined by a phi instruction."""
    for block in function.blocks.values():
        if val_idx in block.phi_defs:
            return True
    return False


def val_as_phi(function: Function, val_idx: int) -> Union[Phi, None]:
    """Returns the Phi instruction that defines the given value (by val_idx), or None if not found."""
    for block in function.blocks.values():
        if val_idx in block.phi_defs:
            # Find the phi instruction that defines this value
            val_name = get_val_name(function, val_idx)
            for phi in block.phis():
                if phi.dest == val_name:
                    return phi
    return None


