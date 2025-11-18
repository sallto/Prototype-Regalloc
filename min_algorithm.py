"""
Min Algorithm for Register Allocation with Spilling and Reloading

Implements the Min algorithm from:
"The Min Algorithm and Local Register Allocation" paper.

The algorithm performs optimal straight-line register allocation by using
next-use distance information to decide which variables to spill when register
pressure exceeds the available number of registers k.
"""

import math
from typing import Dict, List, Set
from dataclasses import dataclass
from ir import Function, Block, Op
from liveness import compute_liveness


@dataclass
class SpillReload:
    """Represents a spill or reload operation."""
    type: str  # "spill" or "reload"
    variable: str
    instruction_idx: int
    block_name: str

    def __str__(self) -> str:
        return f"{self.type} {self.variable} at {self.block_name}:{self.instruction_idx}"


def limit(W: Set[str], S: Set[str], insn_idx: int, block: Block, m: int, spills: List[SpillReload]) -> Set[str]:
    """
    Evict variables from W to keep only m variables with closest next-use distances.

    Args:
        W: Set of variables currently in registers
        S: Set of variables already spilled
        insn_idx: Index of current instruction in block
        block: Block containing the instruction
        m: Maximum number of variables to keep in registers
        spills: List to append spill operations to

    Returns:
        New set W with at most m variables
    """
    if not W or len(W) <= m:
        return W

    # Get next-use distances for variables in W at this instruction
    next_uses = {}
    if block.next_use_by_instr and insn_idx < len(block.next_use_by_instr):
        for var in W:
            if var in block.next_use_by_instr[insn_idx]:
                next_uses[var] = block.next_use_by_instr[insn_idx][var]
            else:
                next_uses[var] = math.inf

    # Sort W by next-use distance (closest first: smallest distance first)
    sorted_vars = sorted(W, key=lambda v: next_uses.get(v, math.inf))

    # Keep only the first m variables, evict the rest
    kept_vars = set(sorted_vars[:m])
    evicted_vars = sorted_vars[m:]

    # Create spills for evicted variables that haven't been spilled before and have finite next use
    for var in evicted_vars:
        next_use_dist = next_uses.get(var, math.inf)
        if var not in S and next_use_dist != math.inf:
            spills.append(SpillReload("spill", var, insn_idx, block.name))

    # Update S: remove evicted variables from the already spilled set
    # (since they're being evicted again, they need to be spilled again)
    S.difference_update(evicted_vars)

    return kept_vars


def limit_last_instruction(W: Set[str], S: Set[str], block: Block, m: int, spills: List[SpillReload]) -> Set[str]:
    """
    Handle the limit call for the last instruction in a block using live_out information.

    For the last instruction, next-use distances are determined by live_out:
    - Variables in live_out have distance = block_len (to reach the block exit)
    - Variables not in live_out have infinite distance

    Args:
        W: Set of variables currently in registers
        S: Set of variables already spilled
        block: Block containing the instruction
        m: Maximum number of variables to keep in registers
        spills: List to append spill operations to

    Returns:
        New set W with at most m variables
    """
    if not W or len(W) <= m:
        return W

    # Get next-use distances for variables in W using live_out
    next_uses = {}
    block_len = len(block.instructions)

    for var in W:
        if isinstance(block.live_out, dict) and var in block.live_out:
            # Variable is live out, next use is at block exit
            next_uses[var] = block.live_out[var] + block_len
        else:
            # Variable is not live out, infinite distance
            next_uses[var] = math.inf

    # Sort W by next-use distance (closest first: smallest distance first)
    sorted_vars = sorted(W, key=lambda v: next_uses.get(v, math.inf))

    # Keep only the first m variables, evict the rest
    kept_vars = set(sorted_vars[:m])
    evicted_vars = sorted_vars[m:]

    # Create spills for evicted variables that haven't been spilled before and have finite next use
    for var in evicted_vars:
        next_use_dist = next_uses.get(var, math.inf)
        if var not in S and next_use_dist != math.inf:
            # Spill before the last instruction
            spills.append(SpillReload("spill", var, len(block.instructions) - 1, block.name))

    # Update S: remove evicted variables from the already spilled set
    S.difference_update(evicted_vars)

    return kept_vars


def min_algorithm(function: Function, k: int = 3) -> Dict[str, List[SpillReload]]:
    """
    Implement the Min algorithm for register allocation with spilling.

    Args:
        function: Function to perform register allocation on
        k: Number of available registers (default 3)

    Returns:
        Dictionary mapping block names to lists of spill/reload operations
    """
    result = {}

    # Process each block independently
    for block_name, block in function.blocks.items():
        spills = []

        # W: variables currently in registers
        # S: variables already spilled (to avoid duplicate spills)
        W = set()
        S = set()

        for insn_idx, instr in enumerate(block.instructions):
            if not isinstance(instr, Op):
                # Skip non-op instructions (jumps, phis)
                continue

            # R = uses that are not already in registers (need reload)
            R = set(instr.uses) - W

            # Add reloaded variables to both W and S
            W.update(R)
            S.update(R)

            # First limit: make room for operands
            W = limit(W, S, insn_idx, block, k, spills)

            # Second limit: make room for results
            # For the last instruction, we need to handle this differently
            if insn_idx < len(block.instructions) - 1:
                # Not the last instruction, use next instruction's index
                W = limit(W, S, insn_idx + 1, block, k - len(instr.defs), spills)
            else:
                # Last instruction: need to handle using live_out information
                # This will be handled in a separate function
                W = limit_last_instruction(W, S, block, k - len(instr.defs), spills)

            # Add newly defined variables to W
            W.update(instr.defs)

            # Create reload operations for variables in R (before the instruction)
            for var in R:
                spills.append(SpillReload("reload", var, insn_idx, block.name))

        result[block_name] = spills

    return result


def test_min_algorithm():
    """Test the Min algorithm on the simple.ir example with k=3 registers."""
    import main

    # Parse the IR file
    ir_file = "examples/reg_pressure/simple.ir"
    print(f"Testing Min algorithm on {ir_file} with k=3 registers")
    print("=" * 60)

    try:
        # Parse the IR file
        with open(ir_file, "r") as f:
            content = f.read()

        # Parse function
        function = main.parse_function(content)
        print("Parsed function successfully!")
        print()

        # Run liveness analysis
        print("Running liveness analysis...")
        compute_liveness(function)
        print("Liveness analysis completed!")
        print()

        # Print the function with liveness info
        main.print_function(function)
        print()

        # Run the Min algorithm
        print("Running Min algorithm with k=3...")
        spills_reloads = min_algorithm(function, k=3)
        print("Min algorithm completed!")
        print()

        # Print results
        print("Spills and Reloads:")
        print("-" * 30)
        total_operations = 0

        for block_name, operations in spills_reloads.items():
            if operations:
                print(f"Block {block_name}:")
                for op in operations:
                    print(f"  {op}")
                    total_operations += 1
                print()
            else:
                print(f"Block {block_name}: No spills or reloads needed")
                print()

        print(f"Total spill/reload operations: {total_operations}")

    except FileNotFoundError:
        print(f"Error: {ir_file} not found")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_min_algorithm()
