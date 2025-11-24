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

from ir import *
import liveness
import min_algorithm
import argparse
import sys



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
    value_idx = 0  # Counter for assigning unique indices to values
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
                # Assign indices to newly defined values
                for def_val in op.defs:
                    if def_val not in function.value_indices:
                        function.value_indices[def_val] = value_idx
                        value_idx += 1
            elif instr_line.startswith("jmp "):
                jmp = parse_jmp_line(instr_line, line_no)
                current_block.instructions.append(jmp)
                current_block.successors = jmp.targets  # Set block successors
            elif instr_line.startswith("phi "):
                phi = parse_phi_line(instr_line, line_no)
                current_block.instructions.append(phi)
                # Assign index to phi destination value
                if phi.dest not in function.value_indices:
                    function.value_indices[phi.dest] = value_idx
                    value_idx += 1
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
        print(f"    Predecessors: {block.predecessors}")
        print(f"    Successors: {block.successors}")
        print(f"    USE set: {sorted(block.use_set) if block.use_set else 'empty'}")
        print(f"    DEF set: {sorted(block.def_set) if block.def_set else 'empty'}")
        print(f"    PhiUses: {sorted(block.phi_uses) if block.phi_uses else 'empty'}")
        print(f"    PhiDefs: {sorted(block.phi_defs) if block.phi_defs else 'empty'}")
        print(f"    Max Register Pressure: {block.max_register_pressure}")
        live_in_str = ', '.join(f"{var}:{dist:.0f}" if dist != float('inf') else f"{var}:inf"
                               for var, dist in sorted(block.live_in.items())) if isinstance(block.live_in, dict) and block.live_in else 'empty'
        live_out_str = ', '.join(f"{var}:{dist:.0f}" if dist != float('inf') else f"{var}:inf"
                                for var, dist in sorted(block.live_out.items())) if isinstance(block.live_out, dict) and block.live_out else 'empty'
        print(f"    LiveIn: {{{live_in_str}}}")
        print(f"    LiveOut: {{{live_out_str}}}")
        print("    Instructions:")
        for i, instr in enumerate(block.instructions):
            if isinstance(instr, Op):
                uses_str = ", ".join(instr.uses) if instr.uses else "none"
                defs_str = ", ".join(instr.defs) if instr.defs else "none"

                # Show next-use distances for values in this instruction
                next_use_str = ""
                if block.next_use_by_instr and i < len(block.next_use_by_instr):
                    next_uses = []
                    for var in sorted(set(instr.uses + instr.defs)):
                        if var in block.next_use_by_instr[i]:
                            dist = block.next_use_by_instr[i][var]
                            dist_str = "inf" if dist == float('inf') else str(int(dist))
                            next_uses.append(f"{var}:{dist_str}")
                    if next_uses:
                        next_use_str = f" next_use={{{', '.join(next_uses)}}}"

                print(f"      {i}: op uses=[{uses_str}] defs=[{defs_str}]{next_use_str}")
            elif isinstance(instr, Jump):
                targets_str = ", ".join(instr.targets)
                print(f"      {i}: jmp {targets_str}")
            elif isinstance(instr, Phi):
                incomings_str = ", ".join(f"{inc.block}={inc.value}" for inc in instr.incomings)
                print(f"      {i}: phi {instr.dest} [{incomings_str}]")
        print()


def test_parser(ir_file: str, k: int = 3) -> None:
    """Test the parser with IR from the specified file."""
    try:
        with open(ir_file, "r") as f:
            content = f.read()

        # Skip the first line if it starts with a comment or if it's README.md
        lines = content.splitlines()
        if lines and (lines[0].startswith("#") or ir_file.endswith("README.md")):
            ir_text = "\n".join(lines[1:])  # Skip the header line
        else:
            ir_text = content

        print(f"Parsing {ir_file} IR...")
        function = parse_function(ir_text)
        print("Parse successful!")
        print()

        # Compute liveness
        print("Computing liveness analysis...")
        liveness.compute_liveness(function)
        print("Liveness analysis completed!")


        # Check liveness correctness
        print("Checking liveness correctness...")
        try:
            liveness.check_liveness_correctness(function)
            print("Liveness correctness check passed!")
        except AssertionError as e:
            print(f"Liveness correctness check failed: {e}")
            return
        print()

        # Run the Min algorithm for register allocation
        print(f"Running Min algorithm with k={k}...")
        spills_reloads = min_algorithm.min_algorithm(function, k=k)
        print("Min algorithm completed!")
        print()

        # Print the IR with spills and reloads
        print(f"IR with Spills and Reloads (k={k}):")
        print("=" * 50)
        min_algorithm.print_function_with_spills(function, spills_reloads)

    except ParseError as e:
        print(f"Parse error: {e}")
    except FileNotFoundError:
        print(f"{ir_file} not found")


def main():
    parser = argparse.ArgumentParser(description="IR Parser and Liveness Analysis")
    parser.add_argument("file", help="Path to the IR file to parse")
    parser.add_argument("-k", "--registers", type=int, default=3,
                        help="Number of available registers (default: 3)")
    args = parser.parse_args()

    test_parser(args.file, args.registers)


if __name__ == "__main__":
    main()
