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
from liveness import get_next_use_distance
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
    val_idx: int  # Value index (integer identifier)
    position: int  # Position where this operation should be inserted (0 = before first instruction)
    block_name: str
    is_coupling: bool = False  # True if this is coupling code between blocks
    edge_info: str = ""  # For coupling operations: "pred->block"

    def __str__(self) -> str:
        location = f"{self.block_name}:{self.position}"
        if self.is_coupling:
            return f"{self.type} val_idx={self.val_idx} at {location} (coupling: {self.edge_info})"
        else:
            return f"{self.type} val_idx={self.val_idx} at {location}"


def insert_spill_reload_sorted(operations_list: List[SpillReload], spill_reload: SpillReload) -> None:
    """
    Insert a SpillReload into a list while maintaining sorted order.
    
    The list is sorted by (position, type_priority, val_idx) where:
    - position: insertion position (0 = before first instruction)
    - type_priority: 0 for "spill", 1 for "reload" (spills come before reloads)
    - val_idx: value index for stable sorting
    
    Args:
        operations_list: List of SpillReload operations (must already be sorted)
        spill_reload: SpillReload operation to insert
    """
    import traceback

    # Compute sort key: (position, type_priority, val_idx)
    type_priority = 0 if spill_reload.type == "spill" else 1
    key = (spill_reload.position, type_priority, spill_reload.val_idx)

    # Find insertion point using binary search
    # Create a key function for comparison
    def get_key(op: SpillReload) -> tuple:
        op_type_priority = 0 if op.type == "spill" else 1
        return (op.position, op_type_priority, op.val_idx)

    keys = [get_key(op) for op in operations_list]
    insert_idx = bisect.bisect_left(keys, key)

    # Insert at the found position
    operations_list.insert(insert_idx, spill_reload)

    # Debug: Check if inserted at the end; if not, print stacktrace
    if insert_idx != len(operations_list) - 1:
        print("[WARN] insert_spill_reload_sorted: Value not inserted at end (insert_idx = {}, list_len = {}). Stacktrace:".format(insert_idx, len(operations_list)))
        traceback.print_stack()



def limit(W: Set[int], S: Set[int], insn_idx: int, block: Block, m: int, spills: List[SpillReload], function: Function) -> Set[int]:
    """
    Evict variables from W to keep only m variables with closest next-use distances.

    Args:
        W: Set of value indices currently in registers
        S: Set of value indices already spilled
        insn_idx: Index of current instruction in block
        block: Block containing the instruction
        m: Maximum number of variables to keep in registers
        spills: List to append spill operations to
        function: The function containing value_indices mapping

    Returns:
        New set W with at most m value indices
    """
    if not W or len(W) <= m:
        return W

    # Sort W by next-use distance (closest first: smallest distance first)
    # get_next_use_distance handles variables defined at current instruction (returns 0)
    sorted_vars = sorted(W, key=lambda v: (get_next_use_distance(block, v, insn_idx, function), v))

    # Keep only the first m variables, evict the rest
    kept_vars = set(sorted_vars[:m])
    evicted_vars = sorted_vars[m:]

    # Create spills for evicted variables that haven't been spilled before and have finite next use
    for val_idx in evicted_vars:
        next_use_dist = get_next_use_distance(block, val_idx, insn_idx, function)
        if val_idx not in S and next_use_dist != math.inf:
            insert_spill_reload_sorted(spills, SpillReload("spill", val_idx, insn_idx, block.name))

    # Update S: remove evicted variables from the already spilled set
    # (since they're being evicted again, they need to be spilled again)
    S.difference_update(evicted_vars)

    return kept_vars




def initUsual(block: Block, pred_W_exits: Dict[str, Set[int]], k: int, function: Function) -> Set[int]:
    """
    Initialize W_entry for a block using the "usual" initialization strategy.

    Counts frequency of variables across predecessors' W_exit sets, then selects
    variables that appear in all predecessors first, followed by others sorted
    by next-use distance at block entry.

    Args:
        block: The block to initialize
        pred_W_exits: Map from predecessor block names to their W_exit sets (value indices)
        k: Number of available registers
        function: The function containing the block

    Returns:
        Set of value indices that should be in registers at block entry
    """
    if not block.predecessors:
        return set()

    freq = defaultdict(int)
    cand = set()
    take = set()

    # Count frequency of each variable across predecessors' W_exit
    # Only consider variables that are actually live-in to this block (by val_idx)
    live_in_val_indices = set()
    if isinstance(block.live_in, dict):
        live_in_val_indices = set(block.live_in.keys())

    for pred_name in block.predecessors:
        if pred_name in pred_W_exits:
            for val_idx in pred_W_exits[pred_name]:
                if val_idx in live_in_val_indices:
                    freq[val_idx] += 1
                    cand.add(val_idx)
        # Variables not in pred_W_exits are ignored (unprocessed predecessors)

    # For phi nodes, ensure incoming values are considered for W_entry
    # Add phi incoming values that are available from their predecessors
    phi_val_indices = set()
    for phi in block.phis():
        for pred_name in block.predecessors:
            incoming_val = phi.incoming_val_for_block(pred_name)
            if incoming_val is not None and incoming_val in function.value_indices:
                incoming_val_idx = function.value_indices[incoming_val]
                if pred_name in pred_W_exits and incoming_val_idx in pred_W_exits[pred_name]:
                    phi_val_indices.add(incoming_val_idx)

    # Add phi vars to candidates with frequency equal to number of predecessors they come from
    # (but actually, for initUsual, we want phi vars that are available)
    for val_idx in phi_val_indices:
        if val_idx not in cand:
            # Count how many predecessors this phi var is available from
            available_from = sum(1 for pred in block.predecessors
                               if pred in pred_W_exits and val_idx in pred_W_exits[pred])
            #assert(available_from == len(block.predecessors))
            if available_from > 0:
                freq[val_idx] = available_from
                cand.add(val_idx)

    # Variables that appear in all predecessors go to take
    num_preds = len(block.predecessors)
    to_remove = []
    for val_idx in cand:
        if freq[val_idx] == num_preds:
            take.add(val_idx)
            to_remove.append(val_idx)

    for val_idx in to_remove:
        cand.remove(val_idx)

    # If we have more variables in take than available registers,
    # select the k best ones from take based on next-use distance
    if len(take) > k:
        # Get next-use distances for take variables using helper function
        first_instr_idx = 0
        sorted_take = sorted(take, key=lambda v: (get_next_use_distance(block, v, first_instr_idx, function), v))
        take = set(sorted_take[:k])

    # Sort remaining candidates by next-use distance at block entry
    if cand:
        # Get next-use distances from the first instruction using helper function
        first_instr_idx = 0
        sorted_cand = sorted(cand, key=lambda v: (get_next_use_distance(block, v, first_instr_idx, function), v))
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


def usedInLoop(loop_header: str, alive_vars: Set[int], loop_membership: Dict[str, Set[str]],
                function: Function) -> Set[int]:
    """
    Return variables from alive_vars that are used in any block within the loop.

    Args:
        loop_header: Name of the loop header
        alive_vars: Set of value indices to check
        loop_membership: Dictionary mapping loop headers to sets of blocks in each loop
        function: The function containing the blocks

    Returns:
        Set of value indices from alive_vars that are used in the loop
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
        for val_idx in alive_vars:
            if val_idx in block.use_set or val_idx in block.phi_uses:
                used_vars.add(val_idx)

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


def sortByNextUse(vars: Set[int], entry_instr_idx: int, block: Block, function: Function) -> List[int]:
    """
    Sort variables by next-use distance from the entry instruction.

    Args:
        vars: Set of value indices to sort
        entry_instr_idx: Index of the entry instruction (usually 0)
        block: Block containing the variables
        function: Function containing value_indices mapping

    Returns:
        List of value indices sorted by next-use distance (closest first)
    """
    def get_next_use_dist(val_idx: int) -> float:
        if hasattr(block, 'live_in') and isinstance(block.live_in, dict) and val_idx in block.live_in:
            return block.live_in[val_idx]
        return float('inf')

    return sorted(vars, key=get_next_use_dist)


def is_variable_defined_at_position(block: Block, var: str, position: int) -> bool:
    """
    Check if a variable is defined at or before a given position in a block.
    
    Args:
        block: The block to check
        var: The variable name to check
        position: Instruction index position (0 = first instruction)
        
    Returns:
        True if the variable is defined at instruction index <= position, False otherwise
    """
    # Check instructions up to and including the position
    for idx in range(min(position + 1, len(block.instructions))):
        instr = block.instructions[idx]
        if isinstance(instr, Op) and var in instr.defs:
            return True
        elif isinstance(instr, Phi) and instr.dest == var:
            # Phi instructions are typically at the beginning of blocks (index 0)
            return True
    return False


def initLoopHeader(block: Block, loop_membership: Dict[str, Set[str]],
                   function: Function, k: int) -> Tuple[Set[int], Set[int]]:
    """
    Initialize W_entry for a loop header block according to the Min algorithm.

    Args:
        block: The loop header block to initialize
        loop_membership: Dictionary mapping loop headers to sets of blocks in each loop
        function: The function containing the block
        k: Number of available registers

    Returns:
        Tuple containing:
        - Set of value indices that should be in registers at block entry
        - Set of value indices that should be spilled before entering the loop
    """
    entry = 0  # Index of the first instruction
    loop = loopOf(block.name, loop_membership)

    # If block is not in a loop, return empty set
    if loop is None:
        return set(), set()

    # alive = block.phis ∪ block.liveIn (already val_idx)
    alive = block.phi_defs.copy()
    if hasattr(block, 'live_in') and isinstance(block.live_in, dict):
        alive.update(block.live_in.keys())

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
        sorted_liveThrough = sortByNextUse(liveThrough, entry, block, function)

        # add = liveThrough[0:freeLoop]
        add = set(sorted_liveThrough[:freeLoop])
        liveThrough = liveThrough - add
    else:
        # sort(cand, entry)
        sorted_cand = sortByNextUse(cand, entry, block, function)

        # cand = cand[0:k]
        cand = set(sorted_cand[:k])

        # add = ∅
        add = set()

    # return cand ∪ add
    return cand | add, liveThrough


def collect_phi_incoming_values(block: Block, pred_name: str, function: Function, pred_W_exit: Set[int]) -> Tuple[Set[int], Set[int]]:
    """
    Collect phi incoming value indices from a specific predecessor and from other predecessors.
    
    Args:
        block: The block containing phi nodes
        pred_name: Name of the predecessor to collect values from
        function: Function containing value_indices mapping
        pred_W_exit: Set of value indices in registers at predecessor exit
        
    Returns:
        Tuple containing:
        - Set of phi incoming value indices from this predecessor that are in pred_W_exit
        - Set of phi incoming value indices from other predecessors
    """
    phi_incoming_val_indices_from_pred = set()
    phi_incoming_val_indices_from_other_preds = set()
    
    for phi in block.phis():
        incoming_val = phi.incoming_val_for_block(pred_name)
        if incoming_val is not None and incoming_val in function.value_indices:
            incoming_val_idx = function.value_indices[incoming_val]
            # If this incoming value comes from this predecessor and is already in that predecessor's W_exit
            if incoming_val_idx in pred_W_exit:
                phi_incoming_val_indices_from_pred.add(incoming_val_idx)
        # Collect incoming values from other predecessors
        for other_pred in block.predecessors:
            if other_pred != pred_name:
                other_incoming_val = phi.incoming_val_for_block(other_pred)
                if other_incoming_val is not None and other_incoming_val in function.value_indices:
                    phi_incoming_val_indices_from_other_preds.add(function.value_indices[other_incoming_val])
    
    return phi_incoming_val_indices_from_pred, phi_incoming_val_indices_from_other_preds


def compute_reload_vars(W_entry: Set[int], pred_W_exit: Set[int], block: Block,
                       vars_available_from_all: Set[int], phi_vals_from_pred: Set[int],
                       phi_vals_from_others: Set[int]) -> Set[int]:
    """
    Compute variables that need to be reloaded at a block boundary.
    
    Args:
        W_entry: Set of value indices that should be in registers at block entry
        pred_W_exit: Set of value indices in registers at predecessor exit
        block: The block being entered
        vars_available_from_all: Variables available from all processed predecessors
        phi_vals_from_pred: Phi incoming values from this predecessor
        phi_vals_from_others: Phi incoming values from other predecessors
        
    Returns:
        Set of value indices that need to be reloaded
    """
    # Reload: All variables in W_entry \ W_exit_pred (excluding phi destinations)
    # But skip variables that are available from all predecessors (they don't need reloads)
    # Also skip phi incoming values that are already available from this specific predecessor
    # And skip phi incoming values that come from other predecessors (not needed from this edge)
    # block.phi_defs and block.def_set are already val_idx
    return ((W_entry - pred_W_exit) - block.phi_defs) - vars_available_from_all - phi_vals_from_pred - phi_vals_from_others - block.def_set


def collect_phi_incoming_values_for_back_edge(block_name: str, succ_block: Block, function: Function) -> Set[int]:
    """
    Collect phi incoming value indices from a block for a successor (used in second pass).
    
    Args:
        block_name: Name of the block providing incoming values
        succ_block: The successor block containing phi nodes
        function: Function containing value_indices mapping
        
    Returns:
        Set of phi incoming value indices from this block
    """
    phi_incoming_val_indices = set()
    for phi in succ_block.phis():
        incoming_val = phi.incoming_val_for_block(block_name)
        if incoming_val is not None and incoming_val in function.value_indices:
            phi_incoming_val_indices.add(function.value_indices[incoming_val])
    return phi_incoming_val_indices


def insert_coupling_code_for_edge(pred_name: str, block_name: str, W_entry: Set[int], S_entry: Set[int],
                                  pred_W_exit: Set[int], pred_S_exit: Set[int], pred_block: Block,
                                  block: Block, function: Function, result: Dict[str, List[SpillReload]],
                                  vars_available_from_all: Set[int], k: int) -> None:
    """
    Insert coupling code (spills/reloads) for an edge between predecessor and block.
    
    This handles:
    1. Reloading variables needed in block but not in predecessor's registers
    2. Spilling variables if reloading would exceed k registers
    3. Spilling variables that are in S_entry but not in pred_S_exit
    
    Args:
        pred_name: Name of the predecessor block
        block_name: Name of the current block
        W_entry: Set of value indices that should be in registers at block entry
        S_entry: Set of value indices spilled at block entry
        pred_W_exit: Set of value indices in registers at predecessor exit
        pred_S_exit: Set of value indices spilled at predecessor exit
        pred_block: The predecessor block object
        block: The current block object
        function: Function containing value_indices mapping
        result: Dictionary mapping block names to lists of SpillReload operations
        vars_available_from_all: Variables available from all processed predecessors
        k: Number of available registers
    """
    # Collect phi incoming values
    phi_vals_from_pred, phi_vals_from_others = collect_phi_incoming_values(block, pred_name, function, pred_W_exit)
    
    # Compute variables that need reloading
    reload_vars = compute_reload_vars(W_entry, pred_W_exit, block, vars_available_from_all,
                                     phi_vals_from_pred, phi_vals_from_others)
    
    # Check if reloading would exceed k registers
    # After reloads, we'll have: pred_W_exit ∪ reload_vars
    W_after_reload = (pred_W_exit | reload_vars) - pred_S_exit
    if len(W_after_reload) > k:
        # Need to spill some variables from pred_W_exit to make room for reloads
        # Compute how many we need to spill
        num_to_spill = len(W_after_reload) - k
        
        # Sort variables in pred_W_exit by next-use distance in the current block
        # Use the first instruction index (0) as reference point
        sorted_pred_vars = sorted(pred_W_exit, key=lambda v: (get_next_use_distance(block, v, 0, function), v))
        
        # Spill the variables with furthest next use (last in sorted list)
        vars_to_spill_for_reloads = set(sorted_pred_vars[-num_to_spill:])
        
        # Insert spills before reloads
        for val_idx in vars_to_spill_for_reloads:
            if get_next_use_distance(block, val_idx, 0, function) != math.inf:
                insert_spill_reload_sorted(result[pred_name], SpillReload("spill", val_idx, len(pred_block.instructions) - 1, pred_name, is_coupling=True, edge_info=f"{pred_name}->{block_name}"))
    
    # Insert reload operations
    for val_idx in reload_vars:
        insert_spill_reload_sorted(result[pred_name], SpillReload("reload", val_idx, len(pred_block.instructions) - 1, pred_name, is_coupling=True, edge_info=f"{pred_name}->{block_name}"))

    # Spill: All variables in (S_entry \ S_exit_pred) ∩ W_exit_pred
    spill_vars = ((S_entry - pred_S_exit) & pred_W_exit)
    for val_idx in spill_vars:
        if get_next_use_distance(block, val_idx, 0, function) != math.inf:
            insert_spill_reload_sorted(result[pred_name], SpillReload("spill", val_idx, len(pred_block.instructions) - 1, pred_name, is_coupling=True, edge_info=f"{pred_name}->{block_name}"))


def min_algorithm(function: Function, loop_membership: Dict[str, Set[str]], k: int = 3) -> Dict[str, List[SpillReload]]:
    """
    Implement the Min algorithm for register allocation with spilling across multiple blocks.

    Processes blocks in topological order, tracks W_exit and S_exit per block,
    computes W_entry and S_entry from predecessors, and inserts coupling code at block borders.

    Args:
        function: Function to perform register allocation on
        loop_membership: Dictionary mapping loop headers to sets of blocks in each loop
        k: Number of available registers (default 3)

    Returns:
        Dictionary mapping block names to lists of spill/reload operations
    """
    # Initialize result dictionary for all blocks
    result = {block_name: [] for block_name in function.blocks.keys()}

    # Get blocks in topological order
    block_order = topological_order(function)

    # Track W_exit and S_exit for each block
    W_exit_map = {}  # block_name -> set of value indices in registers at exit
    S_exit_map = {}  # block_name -> set of value indices spilled at exit

    # Track blocks that need second-pass processing (back edges where predecessor was processed after successor)
    blocks_needing_second_pass = set()

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
            S_exits = set()
            for pred in block.predecessors:
                if pred in S_exit_map:
                    S_exits.update(S_exit_map[pred])
            S_entry = S_exits & W_entry | S_entry

        # Insert coupling code for each predecessor that has already been processed
        # First, compute which variables are available from ALL processed predecessors
        processed_preds = list()
        for pred_name in block.predecessors:
            if pred_name not in W_exit_map:
                blocks_needing_second_pass.add(pred_name)
            else:
                processed_preds.append(pred_name)
        
        vars_available_from_all = set.intersection(*[W_exit_map[p] for p in processed_preds]) if processed_preds else set()

        
        for pred_name in processed_preds:

            pred_W_exit = W_exit_map[pred_name]
            pred_S_exit = S_exit_map.get(pred_name, set())
            pred_block = function.blocks[pred_name]

            # Insert coupling code for this edge
            insert_coupling_code_for_edge(pred_name, block_name, W_entry, S_entry,
                                         pred_W_exit, pred_S_exit, pred_block, block,
                                         function, result, vars_available_from_all, k)

        # Process block instructions starting with W = W_entry, S = S_entry
        W = W_entry
        S = S_entry

        for insn_idx, instr in enumerate(block.instructions):
            # Get uses and defs for this instruction (convert to val_idx)
            if isinstance(instr, Op):
                instr_uses_val_idx = {function.value_indices[var] for var in instr.uses}
                instr_defs_val_idx = {function.value_indices[var] for var in instr.defs}
            elif isinstance(instr, Phi):
                # we can't insert spills or reloads for phi instructions
                W.add(function.value_indices[instr.dest])
                continue
            else:
                continue

            # R = uses that are not already in registers (need reload)
            R = instr_uses_val_idx - W

            # Add reloaded variables to both W and S, and track them
            W.update(R)
            S.update(R)

            # First limit: make room for operands
            W = limit(W, S, insn_idx, block, k, result[block_name], function)

            # Second limit: make room for results
            W = limit(W, S, insn_idx, block, k - len(instr_defs_val_idx), result[block_name], function)


            # Add newly defined variables to W
            W.update(instr_defs_val_idx)

            # Create reload operations for variables in R (before the instruction)
            for val_idx in R:
                insert_spill_reload_sorted(result[block_name], SpillReload("reload", val_idx, insn_idx, block_name))

        # Store exit state for this block
        W_exit_map[block_name] = W.copy()
        S_exit_map[block_name] = S.copy()

    # Second pass: handle coupling code for back edges (loop edges) where predecessor was processed after successor
    for block_name in blocks_needing_second_pass:
        block = function.blocks[block_name]
        block_W_exit = W_exit_map[block_name]

        # For each successor that has phi nodes using values from this block
        for succ_name in block.successors:
            succ_block = function.blocks[succ_name]

            # Collect phi incoming values from this block
            phi_incoming_val_indices = collect_phi_incoming_values_for_back_edge(block_name, succ_block, function)

            # Reload phi incoming values that aren't in registers at this block's exit
            phi_reload_val_indices = phi_incoming_val_indices - block_W_exit
            for val_idx in phi_reload_val_indices:
                # Insert reload at the end of this block (before the jump to successor)
                insert_spill_reload_sorted(result[block_name], SpillReload("reload", val_idx, len(block.instructions) - 1, block_name, is_coupling=True, edge_info=f"{block_name}->{succ_name}"))

    return result


def get_spills_at(spills_reloads: Dict[str, List[SpillReload]], block_name: str, position: int) -> List[SpillReload]:
    """
    Get all spill operations at a given position in a block.
    
    Args:
        spills_reloads: Dictionary mapping block names to lists of SpillReload operations
        block_name: Name of the block
        position: Instruction position (0 = before first instruction)
        
    Returns:
        List of SpillReload operations that are spills at this position
    """
    if block_name not in spills_reloads:
        return []
    
    result = []
    for op in spills_reloads[block_name]:
        if op.type == "spill" and op.position == position:
            result.append(op)
    
    return result


def is_last_use(val_idx: int, block: Block, instr_idx: int, function: Function) -> bool:
    """
    Check if this is the last use of a variable in the block.
    
    A variable is at its last use if:
    1. It's not live-out of the block, OR
    2. It has no more uses after this instruction in the block
    
    Args:
        val_idx: Value index to check
        block: Block containing the instruction
        instr_idx: Index of the current instruction
        function: Function containing the block
        
    Returns:
        True if this is the last use of the variable
    """
    # Check if variable is live-out
    if val_idx in block.live_out:
            # Variable is live-out, so it's not the last use
            return False
    
    # Variable is not live-out, check if there are more uses in this block
    # Scan forward from the next instruction
    # Use next_use_distance utility to check for further uses in the block.
    # next_use_distance returns None if there is no further use, otherwise returns the distance.
    if get_next_use_distance(block, val_idx, instr_idx, function) is not None:
        return False
    
    # No more uses found, this is the last use
    return True


def color_recursive(block_name: str, k: int, color_assignment: Dict[int, int], 
                    assigned: Set[int], dom_tree: Dict[str, List[str]], 
                    function: Function, spills_reloads: Dict[str, List[SpillReload]]) -> None:
    """
    Recursively color variables in a basic block and its dominator tree children.
    
    This implements the SSA-based coloring algorithm that:
    1. Marks colors of live-in variables as occupied
    2. Processes each instruction, freeing colors on last use
    3. Assigns colors to newly defined variables
    4. Recurses on dominator tree children
    
    Args:
        block_name: Name of the block to process
        k: Number of available colors (registers)
        color_assignment: Dictionary mapping value indices to their assigned colors (ρ)
        assigned: Set of currently assigned colors
        dom_tree: Dominator tree mapping blocks to their children
        function: Function containing the blocks
        spills_reloads: Dictionary mapping blocks to spill/reload operations
    """
    block = function.blocks[block_name]
    
    # Reset assigned to only include colors of live-in variables
    # All variables live-in have already been colored (by dominating blocks)
    live_in_val_indices = set(block.live_in.keys())
    
    # Reset assigned set to only include colors of live-in variables
    # But exclude variables that were spilled (they're not in registers)
    assigned.clear()
    
    # Check for variables that were spilled - they don't occupy registers
    spilled_val_indices = set()
    
    # Check spills at position 0 of this block
    spills_at_start = get_spills_at(spills_reloads, block_name, 0)
    spilled_val_indices.update(spill.val_idx for spill in spills_at_start)
    
    # Check if variables were spilled before entering this block
    # If a variable was spilled at the end of any block and not reloaded at position 0,
    # it doesn't occupy a register
    for val_idx in live_in_val_indices:
        # Check if this variable was spilled at the end of any block
        was_spilled_at_end = False
        for other_block_name, other_spills in spills_reloads.items():
            other_block = function.blocks.get(other_block_name)
            if other_block:
                for spill in other_spills:
                    if spill.type == "spill" and spill.val_idx == val_idx:
                        # Check if spill is at the end of the block
                        if spill.position >= len(other_block.instructions):
                            was_spilled_at_end = True
                            break
                        elif spill.position == len(other_block.instructions) - 1:
                            # Spill before last instruction - check if last is a jump
                            if other_block.instructions and isinstance(other_block.instructions[-1], Jump):
                                was_spilled_at_end = True
                                break
                if was_spilled_at_end:
                    break
        
        # Also check spills at position 0 of this block (before first instruction)
        was_spilled_at_start = any(
            spill.val_idx == val_idx for spill in spills_at_start
        )
        
        # If it was spilled at the end of any block or at position 0, check if it's reloaded at position 0
        if was_spilled_at_end or was_spilled_at_start:
            has_reload_at_start = any(
                op.type == "reload" and op.val_idx == val_idx and op.position == 0
                for op in spills_reloads.get(block_name, [])
            )
            if not has_reload_at_start:
                spilled_val_indices.add(val_idx)
    
    # Check for reloads at position 0 - these also indicate variables that were spilled
    reloads_at_start = [op for op in spills_reloads.get(block_name, []) 
                        if op.type == "reload" and op.position == 0]
    spilled_val_indices.update(reload.val_idx for reload in reloads_at_start)
    
    for val_idx in live_in_val_indices:
        if val_idx in color_assignment:
            # Only mark color as occupied if variable wasn't spilled
            # If it was spilled, it will be reloaded and get its color back then
            if val_idx not in spilled_val_indices:
                assigned.add(color_assignment[val_idx])
    
    # Process each instruction in the block
    for i, instr in enumerate(block.instructions):
        # Handle spills at this position - release their colors
        for spill in get_spills_at(spills_reloads, block_name, i):
            if spill.val_idx in color_assignment:
                assigned.discard(color_assignment[spill.val_idx])
        
        # Handle reloads at this position - assign colors to reloaded variables
        reloads_at_pos = [op for op in spills_reloads.get(block_name, []) 
                          if op.type == "reload" and op.position == i]
        for reload in reloads_at_pos:
            if reload.val_idx not in color_assignment:
                # Assign a color to the reloaded variable
                available = set(range(k)) - assigned
                if not available:
                    # No available colors - raise exception
                    available_colors = set(range(k))
                    raise RuntimeError(
                        f"Coloring failed: No available color for reloaded variable val_idx={reload.val_idx} "
                        f"at position {i} in block {block_name}. "
                        f"Available colors: {available_colors}, Assigned colors: {assigned}, "
                        f"k={k}. All {k} registers are occupied."
                    )
                color = min(available)
                color_assignment[reload.val_idx] = color
                assigned.add(color)
            else:
                # Variable already has a color assignment (from previous block)
                # Mark it as assigned
                if reload.val_idx in color_assignment:
                    assigned.add(color_assignment[reload.val_idx])
        
        # Get uses and defs for this instruction (convert to val_idx)
        if isinstance(instr, Op):
            instr_uses_val_idx = {function.value_indices[var] for var in instr.uses if var in function.value_indices}
            instr_defs_val_idx = {function.value_indices[var] for var in instr.defs if var in function.value_indices}
        elif isinstance(instr, Phi):
            # Phi instructions define their destination
            # Uses come from incoming values, but those are handled at block boundaries
            instr_uses_val_idx = set()
            if instr.dest in function.value_indices:
                instr_defs_val_idx = {function.value_indices[instr.dest]}
            else:
                instr_defs_val_idx = set()
        else:
            # Jump instructions don't use or define variables
            instr_uses_val_idx = set()
            instr_defs_val_idx = set()
        
        # For each use, if last use, free the color
        for use_val_idx in instr_uses_val_idx:
            if is_last_use(use_val_idx, block, i, function):
                if use_val_idx in color_assignment:
                    assigned.discard(color_assignment[use_val_idx])
        
        # For each def, assign a color from available colors
        for def_val_idx in instr_defs_val_idx:
            available = set(range(k)) - assigned
            if not available:
                # No available colors - try to free a color from a dead variable or one that will be spilled
                # First, check if any variable will be spilled right after this instruction
                # We can free its color now (but keep it in color_assignment for potential reload)
                spills_after = get_spills_at(spills_reloads, block_name, i + 1)
                for spill in spills_after:
                    if spill.val_idx in color_assignment:
                        color_val = color_assignment[spill.val_idx]
                        if color_val in assigned:
                            # This variable will be spilled right after, free its color now
                            # Keep it in color_assignment in case it's reloaded later
                            assigned.discard(color_val)
                            available.add(color_val)
                            break
                
                # If still no available colors, try to free a color from a dead variable
                if not available:
                    for val_idx, color_val in sorted(color_assignment.items(), key=lambda x: x[1]):
                        if color_val in assigned:
                            is_dead = val_idx not in block.live_out
                            
                            
                            # Also check if there are more uses after current instruction
                            if is_dead:
                                for j in range(i + 1, len(block.instructions)):
                                    next_instr = block.instructions[j]
                                    if isinstance(next_instr, Op):
                                        for use_var in next_instr.uses:
                                            if use_var in function.value_indices and function.value_indices[use_var] == val_idx:
                                                is_dead = False
                                                break
                                    if not is_dead:
                                        break
                            
                            if is_dead:
                                # Free this color - remove from assignment and assigned set
                                del color_assignment[val_idx]
                                assigned.discard(color_val)
                                available.add(color_val)
                                break
            
            if available:
                color = min(available)  # Pick lowest available color
            else:
                # No available colors - this shouldn't happen if spilling was correct
                # Raise an exception to indicate coloring failure
                available_colors = set(range(k))
                raise RuntimeError(
                    f"Coloring failed: No available color for variable val_idx={def_val_idx} "
                    f"at instruction {i} in block {block_name}. "
                    f"Available colors: {available_colors}, Assigned colors: {assigned}, "
                    f"k={k}. All {k} registers are occupied and no dead variables found."
                )
            color_assignment[def_val_idx] = color
            assigned.add(color)
    
    # Recurse on dominator tree children
    for child in dom_tree.get(block_name, []):
        # Pass a copy of assigned so each child gets the correct state
        color_recursive(child, k, color_assignment, assigned.copy(), dom_tree, function, spills_reloads)


def color_program(function: Function, k: int, spills_reloads: Dict[str, List[SpillReload]]) -> Dict[int, int]:
    """
    Color the program using SSA-based register coloring.
    
    This is the entry point for the coloring algorithm. It initializes the color
    assignment dictionary and starts the recursive coloring from the entry block.
    
    Args:
        function: Function to color
        k: Number of available colors (registers)
        spills_reloads: Dictionary mapping blocks to spill/reload operations
        
    Returns:
        Dictionary mapping value indices to their assigned colors (0 to k-1)
    """
    from dominators import find_entry_block, build_dominator_tree, compute_dominators
    
    # Initialize color assignment dictionary (ρ)
    color_assignment: Dict[int, int] = {}
    
    # Find entry block
    entry_block = find_entry_block(function)
    if entry_block is None:
        return color_assignment
    
    # Compute dominator tree
    idom = compute_dominators(function)
    dom_tree = build_dominator_tree(idom)
    
    # Initialize assigned colors set
    assigned: Set[int] = set()
    
    # Start recursive coloring from entry block
    color_recursive(entry_block, k, color_assignment, assigned, dom_tree, function, spills_reloads)
    
    return color_assignment
