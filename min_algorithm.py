"""
Min Algorithm for Register Allocation with Spilling and Reloading

Implements the Min algorithm from:
"The Min Algorithm and Local Register Allocation" paper.

The algorithm performs optimal straight-line register allocation by using
next-use distance information to decide which variables to spill when register
pressure exceeds the available number of registers k.
"""

import math
import bisect
from typing import Dict, List, Set, Union, Tuple
from dataclasses import dataclass
from ir import Function, Block, Op, Jump, Phi
from liveness import compute_liveness, build_loop_forest, get_next_use_distance
from collections import defaultdict

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

@dataclass
class SpillReload:
    """Represents a spill or reload operation."""
    type: str  # "spill" or "reload"
    variable: str
    position: int  # Position where this operation should be inserted (0 = before first instruction)
    block_name: str
    is_coupling: bool = False  # True if this is coupling code between blocks
    edge_info: str = ""  # For coupling operations: "pred->block"

    def __str__(self) -> str:
        location = f"{self.block_name}:{self.position}"
        if self.is_coupling:
            return f"{self.type} {self.variable} at {location} (coupling: {self.edge_info})"
        else:
            return f"{self.type} {self.variable} at {location}"


def insert_spill_reload_sorted(operations_list: List[SpillReload], spill_reload: SpillReload) -> None:
    """
    Insert a SpillReload into a list while maintaining sorted order.
    
    The list is sorted by (position, type_priority, variable) where:
    - position: insertion position (0 = before first instruction)
    - type_priority: 0 for "spill", 1 for "reload" (spills come before reloads)
    - variable: variable name for stable sorting
    
    Args:
        operations_list: List of SpillReload operations (must already be sorted)
        spill_reload: SpillReload operation to insert
    """
    import traceback

    # Compute sort key: (position, type_priority, variable)
    type_priority = 0 if spill_reload.type == "spill" else 1
    key = (spill_reload.position, type_priority, spill_reload.variable)

    # Find insertion point using binary search
    # Create a key function for comparison
    def get_key(op: SpillReload) -> tuple:
        op_type_priority = 0 if op.type == "spill" else 1
        return (op.position, op_type_priority, op.variable)

    keys = [get_key(op) for op in operations_list]
    insert_idx = bisect.bisect_left(keys, key)

    # Insert at the found position
    operations_list.insert(insert_idx, spill_reload)

    # Debug: Check if inserted at the end; if not, print stacktrace
    if insert_idx != len(operations_list) - 1:
        print("[WARN] insert_spill_reload_sorted: Value not inserted at end (insert_idx = {}, list_len = {}). Stacktrace:".format(insert_idx, len(operations_list)))
        traceback.print_stack()



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

    # Sort W by next-use distance (closest first: smallest distance first)
    # get_next_use_distance handles variables defined at current instruction (returns 0)
    sorted_vars = sorted(W, key=lambda v: get_next_use_distance(block, v, insn_idx, function))

    # Keep only the first m variables, evict the rest
    kept_vars = set(sorted_vars[:m])
    evicted_vars = sorted_vars[m:]

    # Create spills for evicted variables that haven't been spilled before and have finite next use
    for var in evicted_vars:
        next_use_dist = get_next_use_distance(block, var, insn_idx, function)
        if var not in S and next_use_dist != math.inf:
            insert_spill_reload_sorted(spills, SpillReload("spill", var, insn_idx, block.name))

    # Update S: remove evicted variables from the already spilled set
    # (since they're being evicted again, they need to be spilled again)
    S.difference_update(evicted_vars)

    return kept_vars




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
            assert(available_from == len(block.predecessors))
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
                   function: Function, k: int) -> Tuple[Set[str], Set[str]]:
    """
    Initialize W_entry for a loop header block according to the Min algorithm.

    Args:
        block: The loop header block to initialize
        loop_membership: Dictionary mapping loop headers to sets of blocks in each loop
        function: The function containing the block
        k: Number of available registers

    Returns:
        Tuple containing:
        - Set of variables that should be in registers at block entry
        - Set of variables that should be spilled before entering the loop
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
        
        if freeLoop < 0:
            freeLoop = 0

        # sort(liveThrough, entry)
        sorted_liveThrough = sortByNextUse(liveThrough, entry, block)

        # add = liveThrough[0:freeLoop]
        add = set(sorted_liveThrough[:freeLoop])
        liveThrough = liveThrough - add
    else:
        # sort(cand, entry)
        sorted_cand = sortByNextUse(cand, entry, block)

        # cand = cand[0:k]
        cand = set(sorted_cand[:k])

        # add = ∅
        add = set()

    # return cand ∪ add
    return cand | add, liveThrough


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
                W_entry, liveThrough = initLoopHeader(block, loop_membership, function, k)
                S_entry = liveThrough
            else:
                # Use initUsual for non-loop headers
                pred_W_exits = {pred: W_exit_map.get(pred, set()) for pred in block.predecessors}
                W_entry = initUsual(block, pred_W_exits, k, function)
                S_entry = set()

            # Compute S_entry: variables spilled on some path to this block or not in W_entry but live_out

            for pred in block.predecessors:
                if pred in S_exit_map:
                    S_entry.update(S_exit_map[pred] & W_entry)
            

        # Insert coupling code for each predecessor that has already been processed
        for pred_name in block.predecessors:
            # Only insert coupling code if the predecessor has been processed
            if pred_name not in W_exit_map:
                continue

            pred_W_exit = W_exit_map[pred_name]
            pred_S_exit = S_exit_map.get(pred_name, set())
            pred_block = function.blocks[pred_name]

            # Reload: All variables in W_entry \ W_exit_pred (excluding phi destinations)
            reload_vars = (W_entry - pred_W_exit) - block.phi_defs
            
            # Check if reloading would exceed k registers
            # After reloads, we'll have: pred_W_exit ∪ reload_vars
            W_after_reload = (pred_W_exit | reload_vars )- pred_S_exit
            if len(W_after_reload) > k:
                # Need to spill some variables from pred_W_exit to make room for reloads
                # Compute how many we need to spill
                num_to_spill = len(W_after_reload) - k
                
                # Sort variables in pred_W_exit by next-use distance in the current block
                # Use the first instruction index (0) as reference point
                sorted_pred_vars = sorted(pred_W_exit, key=lambda v: get_next_use_distance(block, v, 0, function))
                
                # Spill the variables with furthest next use (last in sorted list)
                vars_to_spill_for_reloads = set(sorted_pred_vars[-num_to_spill:])
                
                # Insert spills before reloads
                for var in vars_to_spill_for_reloads:
                    if get_next_use_distance(block, var, 0, function) != math.inf:
                        insert_spill_reload_sorted(result[pred_name], SpillReload("spill", var, len(pred_block.instructions) - 1, pred_name, is_coupling=True, edge_info=f"{pred_name}->{block_name}"))
            
            for var in reload_vars:
                insert_spill_reload_sorted(result[pred_name], SpillReload("reload", var, len(pred_block.instructions) - 1, pred_name, is_coupling=True, edge_info=f"{pred_name}->{block_name}"))

            # Spill: All variables in (S_entry \ S_exit_pred) ∩ W_exit_pred
            spill_vars = ((S_entry - pred_S_exit) & pred_W_exit) #| ((pred_W_exit - pred_S_exit - W_entry) & block.live_in.keys())
            for var in spill_vars:
                if get_next_use_distance(block, var, 0, function) != math.inf:
                    insert_spill_reload_sorted(result[pred_name], SpillReload("spill", var, len(pred_block.instructions) - 1, pred_name, is_coupling=True, edge_info=f"{pred_name}->{block_name}"))

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

            # Add reloaded variables to both W and S, and track them
            W.update(R)
            S.update(R)

            # First limit: make room for operands
            W = limit(W, S, insn_idx, block, k, result[block_name], function)

            # Second limit: make room for results
            W = limit(W, S, insn_idx, block, k - len(instr_defs), result[block_name], function)


            # Add newly defined variables to W
            W.update(instr_defs)

            # Create reload operations for variables in R (before the instruction)
            for var in R:
                insert_spill_reload_sorted(result[block_name], SpillReload("reload", var, insn_idx, block_name))

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
                insert_spill_reload_sorted(result[block_name], SpillReload("reload", var, len(block.instructions) - 1, block_name, is_coupling=True, edge_info=f"{block_name}->{succ_name}"))

    return result
