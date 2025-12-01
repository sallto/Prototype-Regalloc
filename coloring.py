
from ir import Function, Block, Op, Phi
from typing import Dict, List, Set, Optional
from min_algorithm import SpillReload, is_last_use
import dominators


def is_live_in(block: Block, val_idx: int) -> bool:
    """Check if a variable is live-in to a block."""
    return val_idx in block.live_in


def is_live_out_of_op(block: Block, op_idx: int, val_idx: int, function: Function) -> bool:
    """
    Check if a variable is live-out of an operation.
    
    A variable is live-out of an operation if it's not at its last use.
    """
    return not is_last_use(val_idx, block, op_idx, function)


def choose_color(allocated: Set[int], k: int, last_color: Optional[int] = None) -> Optional[int]:
    """
    Choose a free register color using round-robin selection.
    
    If last_color is provided, start checking from (last_color + 1) % k,
    then continue round-robin. Otherwise, start from 0.
    
    Args:
        allocated: Set of register colors currently allocated
        k: Total number of available registers
        last_color: Last assigned color (for round-robin)
        
    Returns:
        First available color in [0, k), or None if all are allocated
    """
    if last_color is None:
        start = 0
    else:
        start = (last_color + 1) % k
    
    # Check from start to k-1
    for i in range(start, k):
        if i not in allocated:
            return i
    
    # Check from 0 to start-1
    for i in range(start):
        if i not in allocated:
            return i
    
    return None


def repair_argument(block: Block, op: Op, op_idx: int, u_val_idx: int, 
                   parallel_copy: List, function: Function, k: int) -> bool:
    """
    Dummy repair function for argument constraints.
    
    Always succeeds for now.
    """
    return True


def repair_result(block: Block, op: Op, op_idx: int, d_val_idx: int,
                 parallel_copy: List, function: Function, k: int) -> bool:
    """
    Dummy repair function for result assignment.
    
    Always succeeds for now.
    """
    return True


def process_operation(block: Block, op: Op, op_idx: int, allocated_variables: Set[int],
                     var_to_color: Dict[int, int], gcolor: Dict[int, int],
                     function: Function, k: int, last_color: Optional[int]) -> Optional[int]:
    """
    Process a single operation according to Algorithm 3.
    
    Args:
        block: The basic block containing the operation
        op: The operation to process
        op_idx: Index of the operation in the block
        allocated_variables: Set of val_idx currently allocated to registers
        var_to_color: Dictionary mapping val_idx to current color (ccolor)
        gcolor: Dictionary mapping val_idx to global color (gcolor)
        function: The function containing the block
        k: Number of available registers
        last_color: Last assigned color (for round-robin)
        
    Returns:
        The last assigned color, or None if no color was assigned
    """
    parallel_copy = []
    dead = set()
    
    # Get set of currently allocated register colors
    allocated_colors = {var_to_color[v] for v in allocated_variables if v in var_to_color}
    
    # Skip phi operations - phi arguments are considered on incoming edges, not here
    if isinstance(op, Phi):
        # Phi operations are handled separately in the main loop
        return last_color
    
    # Check argument constraints and release last used colors
    for u_var in op.uses:
        if u_var not in function.value_indices:
            continue
        
        u_val_idx = function.value_indices[u_var]
        
        # If current color does not match constraints, repair
        # For now, we don't have constraints, so skip this check
        # If we had constraints: if u_val_idx in var_to_color and var_to_color[u_val_idx] not in op.constraints(u_var):
        #     success = repair_argument(block, op, op_idx, u_val_idx, parallel_copy, function, k)
        #     if not success:
        #         # Graph coloring would be called here
        #         return last_color
        
        # Check whether u is last used here or not
        if not is_live_out_of_op(block, op_idx, u_val_idx, function):
            dead.add(u_val_idx)
    
    # Release dead variables
    allocated_variables -= dead
    # Update allocated_colors after removing dead variables
    allocated_colors = {var_to_color[v] for v in allocated_variables if v in var_to_color}
    
    # Assign definitions
    for d_var in op.defs:
        if d_var not in function.value_indices:
            continue
        
        d_val_idx = function.value_indices[d_var]
        
        # Choose color
        color = choose_color(allocated_colors, k, last_color)
        if color is None:
            success = repair_result(block, op, op_idx, d_val_idx, parallel_copy, function, k)
            if not success:
                # Graph coloring would be called here, but we skip for now
                return last_color
            # After repair, try again (recompute allocated_colors)
            allocated_colors = {var_to_color[v] for v in allocated_variables if v in var_to_color}
            color = choose_color(allocated_colors, k, last_color)
            if color is None:
                return last_color
        
        var_to_color[d_val_idx] = color
        gcolor[d_val_idx] = color
        allocated_variables.add(d_val_idx)
        allocated_colors.add(color)
        last_color = color
    
    # Release dead definitions
    for d_var in op.defs:
        if d_var not in function.value_indices:
            continue
        d_val_idx = function.value_indices[d_var]
        if not is_live_out_of_op(block, op_idx, d_val_idx, function):
            allocated_variables.discard(d_val_idx)
            if d_val_idx in var_to_color:
                allocated_colors.discard(var_to_color[d_val_idx])
    
    return last_color


def fix_global_color(block: Block, op_idx: Optional[int], allocated_variables: Set[int],
                    var_to_color: Dict[int, int], gcolor: Dict[int, int],
                    function: Function, k: int):
    """
    Fix global color at the last point of the block where we can insert code.
    
    This is called when op.next is None or op.next.isLateOperation.
    For now, this is a placeholder - the actual implementation would handle
    late operations and edge splitting.
    """
    # Placeholder: update gcolor from var_to_color for all allocated variables
    for val_idx in allocated_variables:
        if val_idx in var_to_color:
            gcolor[val_idx] = var_to_color[val_idx]


def color_program(function: Function, k: int, spills_reloads: Dict[str, List[SpillReload]]) -> Dict[int, int]:
    """
    Tree-scan coloring algorithm (Algorithm 2).
    
    Assigns register colors to SSA values using dominance-based inheritance.
    """
    # Compute dominator tree
    idom = dominators.compute_dominators(function)
    
    # Find entry block
    entry = dominators.find_entry_block(function)
    if entry is None:
        return {}
    
    # Compute reverse post-order traversal
    postorder = dominators.compute_postorder(function, entry)
    reverse_postorder = list(reversed(postorder))
    
    # Global color assignment (returned result)
    gcolor: Dict[int, int] = {}
    
    # Per-block state
    block_allocated_variables: Dict[str, Set[int]] = {}
    block_var_to_color: Dict[str, Dict[int, int]] = {}
    block_last_color: Dict[str, Optional[int]] = {}
    
    # Process blocks in reverse post-order
    for block_name in reverse_postorder:
        block = function.blocks[block_name]
        
        # Initialize set of occupied registers
        if block_name == entry or idom.get(block_name) is None or idom.get(block_name) == block_name:
            # Entry block: start with empty set
            allocated_variables = set()
            var_to_color = {}
            last_color = None
        else:
            # Inherit from immediate dominator
            idom_name = idom[block_name]
            allocated_variables = block_allocated_variables.get(idom_name, set()).copy()
            var_to_color = block_var_to_color.get(idom_name, {}).copy()
            last_color = block_last_color.get(idom_name)
        
        # Filter to only variables that are live-in to this block
        allocated_variables = {v for v in allocated_variables if is_live_in(block, v)}
        # Also filter var_to_color
        var_to_color = {v: c for v, c in var_to_color.items() if is_live_in(block, v)}
        
        # Forward traversal of the operations
        phi_count = 0
        for phi in block.phis():
            phi_count += 1
        
        # Process phi operations first (they're at the beginning)
        # Phi arguments are considered on incoming edges, but we still need to assign colors to phi destinations
        for phi_idx, phi in enumerate(block.phis()):
            # Get set of currently allocated register colors
            allocated_colors = {var_to_color[v] for v in allocated_variables if v in var_to_color}
            
            # Process phi destination
            if phi.dest in function.value_indices:
                d_val_idx = function.value_indices[phi.dest]
                # Choose color for phi destination
                color = choose_color(allocated_colors, k, last_color)
                if color is None:
                    success = repair_result(block, phi, phi_idx, d_val_idx, [], function, k)
                    if not success:
                        # Graph coloring would be called here, but we skip for now
                        pass
                    else:
                        # After repair, try again (recompute allocated_colors)
                        allocated_colors = {var_to_color[v] for v in allocated_variables if v in var_to_color}
                        color = choose_color(allocated_colors, k, last_color)
                
                if color is not None:
                    var_to_color[d_val_idx] = color
                    gcolor[d_val_idx] = color
                    allocated_variables.add(d_val_idx)
                    last_color = color
                    
                    # Release dead definitions
                    if not is_live_out_of_op(block, phi_idx, d_val_idx, function):
                        allocated_variables.discard(d_val_idx)
                        if d_val_idx in var_to_color:
                            allocated_colors.discard(var_to_color[d_val_idx])
        
        # Process remaining operations (non-phi)
        for op_idx, instr in enumerate(block.instructions[phi_count:], start=phi_count):
            if isinstance(instr, Op):
                last_color = process_operation(block, instr, op_idx, allocated_variables,
                                              var_to_color, gcolor, function, k, last_color)
            elif isinstance(instr, Phi):
                # Shouldn't happen after phi_count, but handle it anyway by skipping
                pass
            
            # Check if this is the last point where we can insert code
            # (op.next is None or op.next.isLateOperation)
            # For now, we check if this is the last instruction
            if op_idx == len(block.instructions) - 1:
                fix_global_color(block, op_idx, allocated_variables, var_to_color, gcolor, function, k)
        
        # Store state for this block
        block_allocated_variables[block_name] = allocated_variables.copy()
        block_var_to_color[block_name] = var_to_color.copy()
        block_last_color[block_name] = last_color
    
    return gcolor
