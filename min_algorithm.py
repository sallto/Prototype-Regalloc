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
from ir import Function, Block, Op, Jump, Phi
from liveness import compute_liveness
from collections import defaultdict
from typing import Dict, Set, List


@dataclass
class SpillReload:
    """Represents a spill or reload operation."""
    type: str  # "spill" or "reload"
    variable: str
    instruction_idx: int
    block_name: str
    is_coupling: bool = False  # True if this is coupling code between blocks
    edge_info: str = ""  # For coupling operations: "pred->block"

    def __str__(self) -> str:
        location = f"{self.block_name}:{self.instruction_idx}"
        if self.is_coupling:
            return f"{self.type} {self.variable} at {location} (coupling: {self.edge_info})"
        else:
            return f"{self.type} {self.variable} at {location}"


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

    # Get next-use distances for variables in W from this instruction onwards
    next_uses = {}
    for var in W:
        # Scan forward from insn_idx to find next use of this variable
        next_use_dist = math.inf
        for future_idx in range(insn_idx, len(block.instructions)):
            future_instr = block.instructions[future_idx]
            if isinstance(future_instr, Op) and var in future_instr.uses:
                next_use_dist = future_idx - insn_idx
                break

        # If variable is live out and not used in this block, use the live_out distance
        if next_use_dist == math.inf and hasattr(block, 'live_out') and isinstance(block.live_out, dict) and var in block.live_out:
            # live_out[var] is the distance from block exit, so add distance to exit
            distance_to_exit = len(block.instructions) - insn_idx
            next_use_dist = distance_to_exit + block.live_out[var]

        next_uses[var] = next_use_dist

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
    # idea: store next-use distances from the beginning of the block, i.e. next-use + i,
    # then the current next-use dist is saved-next-use - i.
    if not W or len(W) <= m:
        return W

    # Get next-use distances for variables in W at the last instruction
    next_uses = {}
    last_idx = len(block.instructions) - 1

    for var in W:
        # Check if variable is used in remaining instructions (none, since this is last)
        # If live out, it will be used after the block, so distance = 1
        if hasattr(block, 'live_out') and isinstance(block.live_out, (set, dict)) and var in block.live_out:
            next_uses[var] = 1  # Used right after this instruction (block exit)
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
            # Spill before the last instruction
            spills.append(SpillReload("spill", var, len(block.instructions) - 1, block.name))

    # Update S: remove evicted variables from the already spilled set
    S.difference_update(evicted_vars)

    return kept_vars


def topological_order(function: Function) -> List[str]:
    """
    Return blocks in topological order (predecessors before successors).

    Uses DFS-based topological sort. Entry blocks (no predecessors) come first.

    Args:
        function: The Function object with blocks

    Returns:
        List of block names in topological order
    """
    # Find entry blocks (blocks with no predecessors)
    entry_blocks = [name for name, block in function.blocks.items() if not block.predecessors]

    # If no blocks have predecessors, pick the first block as entry
    if not entry_blocks:
        entry_blocks = [list(function.blocks.keys())[0]]

    visited = set()
    order = []

    def dfs(block_name: str):
        if block_name in visited:
            return
        visited.add(block_name)

        # Visit all successors
        for successor in function.blocks[block_name].successors:
            dfs(successor)

        order.append(block_name)

    # Start DFS from each entry block
    for entry in entry_blocks:
        dfs(entry)

    # Reverse to get topological order (predecessors first)
    return order[::-1]


def initUsual(block: Block, pred_W_exits: Dict[str, Set[str]], k: int, function: Function) -> Set[str]:
    """
    Initialize W_entry for a block using the "usual" initialization strategy.

    Counts frequency of variables across predecessors' W_exit sets, then selects
    variables that appear in all predecessors first, followed by others sorted
    by next-use distance at block entry.

    Args:
        block: The block to initialize
        pred_W_exits: Map from predecessor block names to their W_exit sets
        k: Number of available registers
        function: The function containing the block

    Returns:
        Set of variables that should be in registers at block entry
    """
    if not block.predecessors:
        return set()

    freq = defaultdict(int)
    cand = set()
    take = set()

    # Count frequency of each variable across predecessors' W_exit
    for pred_name in block.predecessors:
        if pred_name in pred_W_exits:
            for var in pred_W_exits[pred_name]:
                freq[var] += 1
                cand.add(var)
        # Variables not in pred_W_exits are ignored (unprocessed predecessors)

    # Variables that appear in all predecessors go to take
    num_preds = len(block.predecessors)
    to_remove = []
    for var in cand:
        if freq[var] == num_preds:
            take.add(var)
            to_remove.append(var)

    for var in to_remove:
        cand.remove(var)

    # If we have more variables in take than available registers,
    # select the k best ones from take based on next-use distance
    if len(take) > k:
        # Get next-use distances for take variables
        next_uses = {}
        first_instr_idx = 0

        for var in take:
            # Scan forward from first instruction to find next use of this variable
            next_use_dist = float('inf')
            for future_idx in range(first_instr_idx, len(block.instructions)):
                future_instr = block.instructions[future_idx]
                if isinstance(future_instr, Op) and var in future_instr.uses:
                    next_use_dist = future_idx - first_instr_idx
                    break
            next_uses[var] = next_use_dist

        # Sort take by next-use distance (closest first) and take top k
        sorted_take = sorted(take, key=lambda v: next_uses.get(v, float('inf')))
        take = set(sorted_take[:k])

    # Sort remaining candidates by next-use distance at block entry
    if cand:
        # Get next-use distances from the first instruction
        next_uses = {}
        first_instr_idx = 0

        for var in cand:
            # Scan forward from first instruction to find next use of this variable
            next_use_dist = float('inf')
            for future_idx in range(first_instr_idx, len(block.instructions)):
                future_instr = block.instructions[future_idx]
                if isinstance(future_instr, Op) and var in future_instr.uses:
                    next_use_dist = future_idx - first_instr_idx
                    break
            next_uses[var] = next_use_dist

        # Sort by next-use distance (closest first)
        sorted_cand = sorted(cand, key=lambda v: next_uses.get(v, float('inf')))
    else:
        sorted_cand = []

    # Return take ∪ first (k - |take|) variables from cand
    remaining_slots = k - len(take)
    selected_cand = set(sorted_cand[:remaining_slots])

    return take | selected_cand


def min_algorithm(function: Function, k: int = 3) -> Dict[str, List[SpillReload]]:
    """
    Implement the Min algorithm for register allocation with spilling across multiple blocks.

    Processes blocks in topological order, tracks W_exit and S_exit per block,
    computes W_entry and S_entry from predecessors, and inserts coupling code at block borders.

    Args:
        function: Function to perform register allocation on
        k: Number of available registers (default 3)

    Returns:
        Dictionary mapping block names to lists of spill/reload operations
    """
    # Initialize result dictionary for all blocks
    result = {block_name: [] for block_name in function.blocks.keys()}

    # Get blocks in topological order
    block_order = topological_order(function)

    # Track W_exit and S_exit for each block
    W_exit_map = {}  # block_name -> set of variables in registers at exit
    S_exit_map = {}  # block_name -> set of variables spilled at exit

    # Process each block in topological order
    for block_name in block_order:
        block = function.blocks[block_name]

        # Initialize W_entry and S_entry for this block
        if not block.predecessors:
            # Entry block: start with empty sets
            W_entry = set()
            S_entry = set()
        else:
            # Compute W_entry from predecessors using initUsual
            pred_W_exits = {pred: W_exit_map.get(pred, set()) for pred in block.predecessors}
            W_entry = initUsual(block, pred_W_exits, k, function)

            # Compute S_entry: variables spilled on some path to this block
            S_entry = set()
            for pred in block.predecessors:
                if pred in S_exit_map:
                    S_entry.update(S_exit_map[pred])
            S_entry &= W_entry  # Only consider variables that are in W_entry

        # Insert coupling code for each predecessor
        for pred_name in block.predecessors:
            pred_W_exit = W_exit_map.get(pred_name, set())
            pred_S_exit = S_exit_map.get(pred_name, set())

            # Reload variables in W_entry \ W_exit_pred on edge pred->block
            reload_vars = W_entry - pred_W_exit
            for var in reload_vars:
                # Insert reload at the beginning of this block (instruction_idx = 0)
                result[block_name].append(SpillReload("reload", var, 0, block_name, is_coupling=True, edge_info=f"{pred_name}->{block_name}"))

            # Spill variables in (S_entry \ S_exit_pred) ∩ W_exit_pred on edge pred->block
            spill_vars = (S_entry - pred_S_exit) & pred_W_exit
            for var in spill_vars:
                # Insert spill at the end of predecessor block
                pred_block = function.blocks[pred_name]
                result[pred_name].append(SpillReload("spill", var, len(pred_block.instructions), pred_name, is_coupling=True, edge_info=f"{pred_name}->{block_name}"))

        # Process block instructions starting with W = W_entry, S = S_entry
        W = W_entry.copy()
        S = S_entry.copy()

        for insn_idx, instr in enumerate(block.instructions):
            # Get uses and defs for this instruction
            if isinstance(instr, Op):
                instr_uses = instr.uses
                instr_defs = instr.defs
            else:
                # Jump and Phi instructions don't use or define variables directly
                instr_uses = []
                instr_defs = []

            # R = uses that are not already in registers (need reload)
            R = set(instr_uses) - W

            # Add reloaded variables to both W and S
            W.update(R)
            S.update(R)

            # First limit: make room for operands
            W = limit(W, S, insn_idx, block, k, result[block_name])

            # Second limit: make room for results
            # For the last instruction, we need to handle this differently
            if insn_idx < len(block.instructions) - 1:
                # Not the last instruction: spills to make room for results should happen before this instruction
                W = limit(W, S, insn_idx, block, k - len(instr_defs), result[block_name])
            else:
                # Last instruction: need to handle using live_out information
                W = limit_last_instruction(W, S, block, k - len(instr_defs), result[block_name])

            # Add newly defined variables to W
            W.update(instr_defs)

            # Create reload operations for variables in R (before the instruction)
            for var in R:
                result[block_name].append(SpillReload("reload", var, insn_idx, block_name))

        # Store exit state for this block
        W_exit_map[block_name] = W.copy()
        S_exit_map[block_name] = S.copy()

    return result


def print_function_with_spills(function: Function, spills_reloads: Dict[str, List[SpillReload]]) -> None:
    """
    Print the function IR with spill and reload instructions inserted at correct positions.

    Args:
        function: The Function object to print
        spills_reloads: Dictionary mapping block names to lists of SpillReload operations
    """
    print(f"function {function.name}")
    print()

    # Process blocks in the order they appear in function.blocks
    for block_name, block in function.blocks.items():
        print(f"block {block_name}:")

        # Get all operations for this block (both coupling and intra-block)
        operations = spills_reloads.get(block_name, [])

        # Separate operations by instruction index
        ops_by_idx = {}
        for op in operations:
            idx = op.instruction_idx
            if idx not in ops_by_idx:
                ops_by_idx[idx] = []
            ops_by_idx[idx].append(op)

        # Sort operations at each index: reloads before spills
        for idx in ops_by_idx:
            ops_by_idx[idx].sort(key=lambda x: (0 if x.type == "reload" else 1, x.variable))

        # Build the instruction sequence with spills/reloads inserted

        # Process each original instruction
        for instr_idx, instr in enumerate(block.instructions):
            # Handle operations before this instruction (reloads and spills from previous limit calls)
            if instr_idx in ops_by_idx:
                for op in ops_by_idx[instr_idx]:
                    print(f"  {op.type} {op.variable}")

            # Print the original instruction
            if isinstance(instr, Op):
                uses_str = ",".join(instr.uses) if instr.uses else ""
                defs_str = ",".join(instr.defs) if instr.defs else ""
                parts = []
                if uses_str:
                    parts.append(f"uses={uses_str}")
                if defs_str:
                    parts.append(f"defs={defs_str}")
                op_line = "op"
                if parts:
                    op_line += " " + " ".join(parts)
                print(f"  {op_line}")
            elif isinstance(instr, Jump):
                targets_str = ",".join(instr.targets)
                print(f"  jmp {targets_str}")
            elif isinstance(instr, Phi):
                incomings_str = ", ".join(f"{inc.block}, {inc.value}" for inc in instr.incomings)
                print(f"  phi {instr.dest} [{incomings_str}]")

        # Handle operations after the last instruction (coupling spills)
        last_idx = len(block.instructions)
        if last_idx in ops_by_idx:
            for op in ops_by_idx[last_idx]:
                print(f"  {op.type} {op.variable}")

        print()


def test_min_algorithm():
    """Test the Min algorithm on the simple.ir example with k=3 registers."""
    import main

    # Parse the IR file
    ir_file = "examples/reg_pressure/linear.ir"
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
        print("=" * 50)

        # Separate coupling and intra-block operations
        coupling_ops = []
        intra_ops = []

        for block_name, operations in spills_reloads.items():
            for op in operations:
                if op.is_coupling:
                    coupling_ops.append(op)
                else:
                    intra_ops.append(op)

        # Print coupling operations (block boundary spills/reloads)
        if coupling_ops:
            print("Coupling Operations (Block Boundaries):")
            print("-" * 40)
            for op in sorted(coupling_ops, key=lambda x: (x.block_name, x.instruction_idx)):
                print(f"  {op}")
            print()

        # Print intra-block operations
        print("Intra-Block Operations:")
        print("-" * 25)
        total_intra = 0
        total_coupling = len(coupling_ops)

        for block_name in sorted(function.blocks.keys()):
            operations = spills_reloads[block_name]
            intra_block_ops = [op for op in operations if not op.is_coupling]

            if intra_block_ops:
                print(f"Block {block_name}:")
                for op in sorted(intra_block_ops, key=lambda x: x.instruction_idx):
                    print(f"  {op}")
                    total_intra += 1
                print()
            else:
                print(f"Block {block_name}: No intra-block spills or reloads")
                print()

        print("=" * 50)
        print(f"Summary: {total_coupling} coupling operations, {total_intra} intra-block operations")
        print(f"Total operations: {total_coupling + total_intra}")
        print()

        # Print the IR with spills and reloads inserted
        print("IR with Spills and Reloads:")
        print("=" * 50)
        print_function_with_spills(function, spills_reloads)

    except FileNotFoundError:
        print(f"Error: {ir_file} not found")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_min_algorithm()
