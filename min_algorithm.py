"""
Min Algorithm for Register Allocation with Spilling and Reloading

Implements the Min algorithm from:
"The Min Algorithm and Local Register Allocation" paper.

The algorithm performs optimal straight-line register allocation by using
next-use distance information to decide which variables to spill when register
pressure exceeds the available number of registers k.
"""

import math
from typing import Dict, List, Set, Union
from dataclasses import dataclass
from ir import Function, Block, Op, Jump, Phi
from liveness import compute_liveness, build_loop_forest, get_next_use_distance
from collections import defaultdict


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


def limit(W: Set[str], S: Set[str], insn_idx: int, block: Block, m: int, spills: List[SpillReload], function: Function) -> Set[str]:
    """
    Evict variables from W to keep only m variables with closest next-use distances.

    Args:
        W: Set of variables currently in registers
        S: Set of variables already spilled
        insn_idx: Index of current instruction in block
        block: Block containing the instruction
        m: Maximum number of variables to keep in registers
        spills: List to append spill operations to
        function: The function containing value_indices mapping

    Returns:
        New set W with at most m variables
    """
    if not W or len(W) <= m:
        return W

    # Get next-use distances for variables in W from this instruction onwards
    next_uses = {}
    for var in W:
        # Check if variable is defined at the current instruction
        current_instr = block.instructions[insn_idx] if insn_idx < len(block.instructions) else None
        is_defined_here = (current_instr and isinstance(current_instr, Op) and var in current_instr.defs)

        if is_defined_here:
            # Variable is redefined at this instruction, so its current value will be overwritten
            # Don't spill it before the redefinition
            next_use_dist = math.inf
        else:
            # Use the helper function to get next-use distance from the new data structure
            next_use_dist = get_next_use_distance(block, var, insn_idx, function)

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

    # For phi nodes, ensure incoming values are considered for W_entry
    # Add phi incoming values that are available from their predecessors
    phi_vars = set()
    for instr in block.instructions:
        if isinstance(instr, Phi):
            for incoming in instr.incomings:
                if incoming.block in pred_W_exits and incoming.value in pred_W_exits[incoming.block]:
                    phi_vars.add(incoming.value)

    # Add phi vars to candidates with frequency equal to number of predecessors they come from
    # (but actually, for initUsual, we want phi vars that are available)
    for var in phi_vars:
        if var not in cand:
            # Count how many predecessors this phi var is available from
            available_from = sum(1 for pred in block.predecessors
                               if pred in pred_W_exits and var in pred_W_exits[pred])
            if available_from > 0:
                freq[var] = available_from
                cand.add(var)

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
        # Get next-use distances for take variables using helper function
        first_instr_idx = 0
        sorted_take = sorted(take, key=lambda v: get_next_use_distance(block, v, first_instr_idx, function))
        take = set(sorted_take[:k])

    # Sort remaining candidates by next-use distance at block entry
    if cand:
        # Get next-use distances from the first instruction using helper function
        first_instr_idx = 0
        sorted_cand = sorted(cand, key=lambda v: get_next_use_distance(block, v, first_instr_idx, function))
    else:
        sorted_cand = []

    # Return take ∪ first (k - |take|) variables from cand
    remaining_slots = k - len(take)
    selected_cand = set(sorted_cand[:remaining_slots])

    return take | selected_cand


def loopOf(block_name: str, loop_membership: Dict[str, Set[str]]) -> Union[str, None]:
    """
    Find which loop header (if any) contains the given block.

    Args:
        block_name: Name of the block to check
        loop_membership: Dictionary mapping loop headers to sets of blocks in each loop

    Returns:
        Loop header name if block is in a loop, None otherwise
    """
    for loop_header, loop_blocks in loop_membership.items():
        if block_name in loop_blocks:
            return loop_header
    return None


def usedInLoop(loop_header: str, alive_vars: Set[str], loop_membership: Dict[str, Set[str]],
                function: Function) -> Set[str]:
    """
    Return variables from alive_vars that are used in any block within the loop.

    Args:
        loop_header: Name of the loop header
        alive_vars: Set of variables to check
        loop_membership: Dictionary mapping loop headers to sets of blocks in each loop
        function: The function containing the blocks

    Returns:
        Set of variables from alive_vars that are used in the loop
    """
    if loop_header not in loop_membership:
        return set()

    used_vars = set()
    loop_blocks = loop_membership[loop_header]

    for block_name in loop_blocks:
        if block_name not in function.blocks:
            continue
        block = function.blocks[block_name]

        # Check if any alive variables are used in this block
        for var in alive_vars:
            if var in block.use_set or var in block.phi_uses:
                used_vars.add(var)

    return used_vars


def getLoopMaxPressure(loop_header: str, loop_membership: Dict[str, Set[str]],
                       function: Function) -> int:
    """
    Compute maximum register pressure across all blocks in the loop.

    Args:
        loop_header: Name of the loop header
        loop_membership: Dictionary mapping loop headers to sets of blocks in each loop
        function: The function containing the blocks

    Returns:
        Maximum register pressure across all blocks in the loop, or 0 if no blocks
    """
    if loop_header not in loop_membership:
        return 0

    max_pressure = 0
    loop_blocks = loop_membership[loop_header]

    for block_name in loop_blocks:
        if block_name in function.blocks:
            max_pressure = max(max_pressure, function.blocks[block_name].max_register_pressure)

    return max_pressure


def sortByNextUse(vars: Set[str], entry_instr_idx: int, block: Block) -> List[str]:
    """
    Sort variables by next-use distance from the entry instruction.

    Args:
        vars: Set of variables to sort
        entry_instr_idx: Index of the entry instruction (usually 0)
        block: Block containing the variables

    Returns:
        List of variables sorted by next-use distance (closest first)
    """
    def get_next_use_dist(var: str) -> float:
        if hasattr(block, 'live_in') and isinstance(block.live_in, dict) and var in block.live_in:
            return block.live_in[var]
        else:
            return float('inf')

    return sorted(vars, key=get_next_use_dist)


def initLoopHeader(block: Block, loop_membership: Dict[str, Set[str]],
                   function: Function, k: int) -> Set[str]:
    """
    Initialize W_entry for a loop header block according to the Min algorithm.

    Args:
        block: The loop header block to initialize
        loop_membership: Dictionary mapping loop headers to sets of blocks in each loop
        function: The function containing the block
        k: Number of available registers

    Returns:
        Set of variables that should be in registers at block entry
    """
    entry = 0  # Index of the first instruction
    loop = loopOf(block.name, loop_membership)

    # If block is not in a loop, return empty set
    if loop is None:
        return set()

    # alive = block.phis ∪ block.liveIn
    alive = block.phi_defs | set(block.live_in.keys()) if hasattr(block, 'live_in') and isinstance(block.live_in, dict) else block.phi_defs

    # cand = usedInLoop(loop, alive)
    cand = usedInLoop(loop, alive, loop_membership, function)

    # liveThrough = alive \ cand
    liveThrough = alive - cand
    print(f"liveThrough: {liveThrough}")

    if len(cand) < k:
        # freeLoop = k − loop.maxPressure + |liveThrough|
        max_pressure = getLoopMaxPressure(loop, loop_membership, function)
        print(f"max_pressure: {max_pressure}")
        freeLoop = k - max_pressure + len(liveThrough)

        # sort(liveThrough, entry)
        sorted_liveThrough = sortByNextUse(liveThrough, entry, block)

        # add = liveThrough[0:freeLoop]
        add = set(sorted_liveThrough[:freeLoop])
    else:
        # sort(cand, entry)
        sorted_cand = sortByNextUse(cand, entry, block)

        # cand = cand[0:k]
        cand = set(sorted_cand[:k])

        # add = ∅
        add = set()

    # return cand ∪ add
    return cand | add


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

    # Build loop membership information
    _, _, loop_membership = build_loop_forest(function)

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
            # Check if this block is a loop header
            if block_name in loop_membership:
                # Use initLoopHeader for loop headers
                W_entry = initLoopHeader(block, loop_membership, function, k)
            else:
                # Use initUsual for non-loop headers
                pred_W_exits = {pred: W_exit_map.get(pred, set()) for pred in block.predecessors}
                W_entry = initUsual(block, pred_W_exits, k, function)

            # Compute S_entry: variables spilled on some path to this block
            S_entry = set()
            for pred in block.predecessors:
                if pred in S_exit_map:
                    S_entry.update(S_exit_map[pred])
            S_entry &= W_entry  # Only consider variables that are in W_entry

        # Insert coupling code for each predecessor that has already been processed
        for pred_name in block.predecessors:
            # Only insert coupling code if the predecessor has been processed
            if pred_name not in W_exit_map:
                continue

            pred_W_exit = W_exit_map[pred_name]
            pred_S_exit = S_exit_map.get(pred_name, set())

            # For phi nodes, ensure incoming values are available at block boundary
            # Only reload phi incoming values that come from this specific predecessor
            phi_incoming_vars = set()
            for instr in block.instructions:
                if isinstance(instr, Phi):
                    for incoming in instr.incomings:
                        if incoming.block == pred_name:
                            phi_incoming_vars.add(incoming.value)
            # Reload variables needed for phi incoming values that aren't in registers
            phi_reload_vars = phi_incoming_vars - pred_W_exit
            for var in phi_reload_vars:
                # Insert reload before the last instruction in predecessor block
                pred_block = function.blocks[pred_name]
                result[pred_name].append(SpillReload("reload", var, len(pred_block.instructions) - 1, pred_name, is_coupling=True, edge_info=f"{pred_name}->{block_name}"))

            # Reload variables in (W_entry ∩ vars_available_from_pred) \ W_exit_pred on edge pred->block
            # (excluding phi destinations and phi incoming values from this predecessor)
            # Phi incoming values are handled separately above

            # Variables that could potentially be reloaded from this predecessor:
            # - Variables defined in the predecessor block (available at exit)
            # - Variables that are live into the predecessor (but these would need to be reloaded even earlier)
            pred_block = function.blocks[pred_name]
            vars_available_from_pred = set()

            # Variables defined in predecessor are available at its exit
            for instr in pred_block.instructions:
                if isinstance(instr, Op) and instr.defs:
                    vars_available_from_pred.update(instr.defs)
                elif isinstance(instr, Phi) and instr.dest:
                    vars_available_from_pred.add(instr.dest)

            # Variables that are live into the predecessor might also be available
            # (though they would need to be reloaded in an earlier predecessor)
            if hasattr(pred_block, 'live_in') and isinstance(pred_block.live_in, dict):
                vars_available_from_pred.update(pred_block.live_in.keys())

            # Only reload variables that are both in W_entry AND available from this predecessor
            all_phi_incoming_vars = set()
            for instr in block.instructions:
                if isinstance(instr, Phi):
                    for incoming in instr.incomings:
                        all_phi_incoming_vars.add(incoming.value)

            reload_vars = (W_entry & vars_available_from_pred - pred_W_exit) - block.phi_defs - all_phi_incoming_vars
            for var in reload_vars:
                # Insert reload before the last instruction in predecessor block
                result[pred_name].append(SpillReload("reload", var, len(pred_block.instructions) - 1, pred_name, is_coupling=True, edge_info=f"{pred_name}->{block_name}"))

            # Spill variables in (S_entry \ S_exit_pred) ∩ W_exit_pred on edge pred->block
            spill_vars = (S_entry - pred_S_exit) & pred_W_exit
            for var in spill_vars:
                # Insert spill before the last instruction (typically the jump) in predecessor block
                pred_block = function.blocks[pred_name]
                result[pred_name].append(SpillReload("spill", var, len(pred_block.instructions) - 1, pred_name, is_coupling=True, edge_info=f"{pred_name}->{block_name}"))

        # Process block instructions starting with W = W_entry, S = S_entry
        W = W_entry.copy()
        S = S_entry.copy()

        for insn_idx, instr in enumerate(block.instructions):
            # Get uses and defs for this instruction
            if isinstance(instr, Op):
                instr_uses = instr.uses
                instr_defs = instr.defs
            elif isinstance(instr, Phi):
                # Phi instructions define their destination but don't use variables directly
                # (uses come from incoming values, handled at block boundaries)
                instr_uses = []
                instr_defs = [instr.dest]
            else:
                # Jump instructions don't use or define variables directly
                instr_uses = []
                instr_defs = []

            # R = uses that are not already in registers (need reload)
            R = set(instr_uses) - W

            # Add reloaded variables to both W and S
            W.update(R)
            S.update(R)

            # First limit: make room for operands
            W = limit(W, S, insn_idx, block, k, result[block_name], function)

            # Second limit: make room for results
            # For the last instruction, we need to handle this differently
            W = limit(W, S, insn_idx, block, k - len(instr_defs), result[block_name], function)


            # Add newly defined variables to W
            W.update(instr_defs)

            # Create reload operations for variables in R (before the instruction)
            for var in R:
                result[block_name].append(SpillReload("reload", var, insn_idx, block_name))

        # Check if we need to spill variables before entering loops with high register pressure
        for succ_name in block.successors:
            if succ_name in loop_membership:
                # This successor is a loop header - check if loop has high register pressure
                max_pressure = getLoopMaxPressure(succ_name, loop_membership, function)
                if max_pressure > k:
                    # Loop has high register pressure - spill variables not used in the loop
                    succ_block = function.blocks[succ_name]
                    # Compute the alive variables at loop header entry (same as initLoopHeader)
                    alive_vars = succ_block.phi_defs | set(succ_block.live_in.keys()) if hasattr(succ_block, 'live_in') and isinstance(succ_block.live_in, dict) else succ_block.phi_defs
                    used_in_loop = usedInLoop(succ_name, alive_vars, loop_membership, function)

                    # Also consider phi incoming values that flow into used phi destinations
                    phi_used_vars = set()
                    for instr in succ_block.instructions:
                        if isinstance(instr, Phi) and instr.dest in used_in_loop:
                            for incoming in instr.incomings:
                                phi_used_vars.add(incoming.value)
                    used_in_loop.update(phi_used_vars)

                    # Only consider variables that are live out of this block for spilling
                    live_out_vars = set(block.live_out.keys()) if hasattr(block, 'live_out') and isinstance(block.live_out, dict) else set()
                    # Variables in W that are live out but not used in the loop should be spilled
                    vars_to_spill = ((W & live_out_vars) - used_in_loop) - block.phi_defs

                    # Spill these variables at the last instruction (before jump to loop)
                    last_idx = len(block.instructions) - 1
                    for var in vars_to_spill:
                        if var not in S:  # Only spill if not already spilled
                            result[block_name].append(SpillReload("spill", var, last_idx, block_name))
                            W.remove(var)
                            S.add(var)

        # Store exit state for this block
        W_exit_map[block_name] = W.copy()
        S_exit_map[block_name] = S.copy()

    # Second pass: handle coupling code for back edges (loop edges) where predecessor was processed after successor
    for block_name in block_order:
        block = function.blocks[block_name]
        block_W_exit = W_exit_map[block_name]

        # For each successor that has phi nodes using values from this block
        for succ_name in block.successors:
            if succ_name not in function.blocks:
                continue

            succ_block = function.blocks[succ_name]

            # Check if this successor has phi nodes that need incoming values from this block
            phi_incoming_vars = set()
            for instr in succ_block.instructions:
                if isinstance(instr, Phi):
                    for incoming in instr.incomings:
                        if incoming.block == block_name:
                            phi_incoming_vars.add(incoming.value)

            # Reload phi incoming values that aren't in registers at this block's exit
            phi_reload_vars = phi_incoming_vars - block_W_exit
            for var in phi_reload_vars:
                # Insert reload at the end of this block (before the jump to successor)
                result[block_name].append(SpillReload("reload", var, len(block.instructions) - 1, block_name, is_coupling=True, edge_info=f"{block_name}->{succ_name}"))

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

        # Sort operations at each index: spills before reloads
        for idx in ops_by_idx:
            ops_by_idx[idx].sort(key=lambda x: (0 if x.type == "spill" else 1, x.variable))

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
    ir_file = "examples/reg_pressure/loop.ir"
    print(f"Testing Min algorithm on {ir_file} with k=3 registers")
    print("=" * 60)

    try:
        # Parse the IR file
        with open(ir_file, "r") as f:
            content = f.read()

        # Parse function (skip comments)
        lines = content.split('\n')
        # Skip comment lines at the beginning
        start_idx = 0
        while start_idx < len(lines) and lines[start_idx].strip().startswith('#'):
            start_idx += 1
        content = '\n'.join(lines[start_idx:])
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
