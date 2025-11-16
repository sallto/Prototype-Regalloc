"""
IR Parser for Simple SSA-like Intermediate Representation

Grammar:
- Function header: "function <name>" (no colon)
- Blank lines allowed between sections
- Block header: "block <name>:" (with colon)
- Instructions are indented by exactly two spaces
- op instructions: "op" followed by optional "uses=%v0,%v1" and/or "defs=%v2,%v3" clauses
  - Multiple uses/defs can appear on the same line, separated by spaces
  - Values are comma-separated with no spaces around commas
- jmp instructions: "jmp <block>[,<block>...]" (comma-separated targets, no spaces)
- phi instructions: "phi <dest> [<block>, <value>, <block>, <value> ...]"
  - Bracket contents are comma+space-separated pairs
- Identifiers: % followed by letters, numbers, underscores (e.g., %v0, %result_1)
"""

from dataclasses import dataclass
from typing import List, Dict


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

    def __init__(self, uses: List[str] = None, defs: List[str] = None):
        super().__init__(kind="op")
        self.uses = uses or []
        self.defs = defs or []


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

    def __init__(self, name: str):
        self.name = name
        self.instructions = []
        self.successors = []


@dataclass
class Function:
    name: str
    blocks: Dict[str, Block]

    def __init__(self, name: str):
        self.name = name
        self.blocks = {}

    def add_block(self, block: Block) -> None:
        self.blocks[block.name] = block


def parse_function(text: str) -> Function:
    """Parse IR text into a Function object."""
    lines = text.splitlines()
    line_no = 0

    # Skip empty lines at the beginning
    while line_no < len(lines) and lines[line_no].strip() == "":
        line_no += 1

    if line_no >= len(lines):
        raise ParseError("Empty input", line_no + 1)

    # Parse function header
    func_line = lines[line_no].strip()
    if not func_line.startswith("function "):
        raise ParseError(f"Expected 'function <name>', got '{func_line}'", line_no + 1)
    func_name = func_line[len("function "):].strip()
    if not func_name:
        raise ParseError("Function name cannot be empty", line_no + 1)

    function = Function(func_name)
    current_block = None
    line_no += 1

    while line_no < len(lines):
        line = lines[line_no]
        line_no += 1

        # Skip empty lines
        if line.strip() == "":
            continue

        # Check for block header
        if line.startswith("block "):
            if not line.endswith(":"):
                raise ParseError(f"Block header must end with ':', got '{line}'", line_no)
            block_name = line[len("block "):-1].strip()
            if not block_name:
                raise ParseError("Block name cannot be empty", line_no)
            if block_name in function.blocks:
                raise ParseError(f"Duplicate block name '{block_name}'", line_no)

            current_block = Block(block_name)
            function.add_block(current_block)
            continue

        # Check for instruction (must be indented by exactly two spaces)
        if line.startswith("  ") and not line.startswith("   "):
            if current_block is None:
                raise ParseError("Instruction found before any block", line_no)

            instr_line = line[2:]  # Remove the two spaces
            if instr_line.startswith("op "):
                op = parse_op_line(instr_line, line_no)
                current_block.instructions.append(op)
            elif instr_line.startswith("jmp "):
                jmp = parse_jmp_line(instr_line, line_no)
                current_block.instructions.append(jmp)
                current_block.successors = jmp.targets  # Set block successors
            elif instr_line.startswith("phi "):
                phi = parse_phi_line(instr_line, line_no)
                current_block.instructions.append(phi)
            else:
                raise ParseError(f"Unknown instruction type: '{instr_line}'", line_no)
            continue

        # If we get here, the line is malformed
        raise ParseError(f"Unexpected line format: '{line}'", line_no)

    if not function.blocks:
        raise ParseError("Function must have at least one block", len(lines))

    return function


def parse_op_line(line: str, line_no: int) -> Op:
    """Parse an op instruction line like 'op uses=%v0 defs=%v1'."""
    parts = line.split()
    if len(parts) < 1 or parts[0] != "op":
        raise ParseError(f"Expected 'op', got '{line}'", line_no)

    uses = []
    defs = []

    for part in parts[1:]:
        if part.startswith("uses="):
            uses_str = part[len("uses="):]
            if not uses_str:
                raise ParseError("Empty uses list", line_no)
            uses = uses_str.split(",")
        elif part.startswith("defs="):
            defs_str = part[len("defs="):]
            if not defs_str:
                raise ParseError("Empty defs list", line_no)
            defs = defs_str.split(",")
        else:
            raise ParseError(f"Unexpected op parameter: '{part}'", line_no)

    return Op(uses=uses, defs=defs)


def parse_jmp_line(line: str, line_no: int) -> Jump:
    """Parse a jmp instruction line like 'jmp b1,b2'."""
    parts = line.split()
    if len(parts) != 2 or parts[0] != "jmp":
        raise ParseError(f"Expected 'jmp <targets>', got '{line}'", line_no)

    targets_str = parts[1]
    if not targets_str:
        raise ParseError("Jump targets cannot be empty", line_no)

    targets = targets_str.split(",")
    if not all(targets):
        raise ParseError("Jump targets cannot be empty", line_no)

    return Jump(targets=targets)


def parse_phi_line(line: str, line_no: int) -> Phi:
    """Parse a phi instruction line like 'phi %v6 [b0, %v1, b1, %v4]'."""
    parts = line.split()
    if len(parts) < 3 or parts[0] != "phi":
        raise ParseError(f"Expected 'phi <dest> [<incomings>]', got '{line}'", line_no)

    dest = parts[1]
    if not dest.startswith("%"):
        raise ParseError(f"Phi destination must start with '%', got '{dest}'", line_no)

    # Find the bracket content
    bracket_start = line.find("[")
    bracket_end = line.find("]")
    if bracket_start == -1 or bracket_end == -1 or bracket_end < bracket_start:
        raise ParseError("Phi instruction missing brackets", line_no)

    bracket_content = line[bracket_start + 1:bracket_end].strip()
    if not bracket_content:
        raise ParseError("Phi incoming list cannot be empty", line_no)

    # Parse comma+space separated pairs
    items = [item.strip() for item in bracket_content.split(", ")]
    if len(items) % 2 != 0:
        raise ParseError("Phi incoming list must have even number of items (block,value pairs)", line_no)

    incomings = []
    for i in range(0, len(items), 2):
        block = items[i]
        value = items[i + 1]
        if not value.startswith("%"):
            raise ParseError(f"Phi value must start with '%', got '{value}'", line_no)
        incomings.append(PhiIncoming(block=block, value=value))

    return Phi(dest=dest, incomings=incomings)


def print_function(function: Function) -> None:
    """Pretty-print a parsed Function."""
    print(f"Function: {function.name}")
    print("Blocks:")
    for block_name, block in function.blocks.items():
        print(f"  {block_name}:")
        print(f"    Successors: {block.successors}")
        print("    Instructions:")
        for i, instr in enumerate(block.instructions):
            if isinstance(instr, Op):
                uses_str = ", ".join(instr.uses) if instr.uses else "none"
                defs_str = ", ".join(instr.defs) if instr.defs else "none"
                print(f"      {i}: op uses=[{uses_str}] defs=[{defs_str}]")
            elif isinstance(instr, Jump):
                targets_str = ", ".join(instr.targets)
                print(f"      {i}: jmp {targets_str}")
            elif isinstance(instr, Phi):
                incomings_str = ", ".join(f"{inc.block}={inc.value}" for inc in instr.incomings)
                print(f"      {i}: phi {instr.dest} [{incomings_str}]")
        print()


def test_parser() -> None:
    """Test the parser with the example from README.md."""
    try:
        with open("README.md", "r") as f:
            content = f.read()

        # Find the IR content (everything after the first line)
        lines = content.splitlines()
        ir_text = "\n".join(lines[1:])  # Skip the "# Basic IR" header

        print("Parsing README.md IR...")
        function = parse_function(ir_text)
        print("Parse successful!")
        print()
        print_function(function)

    except ParseError as e:
        print(f"Parse error: {e}")
    except FileNotFoundError:
        print("README.md not found")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    test_parser()


if __name__ == "__main__":
    main()
