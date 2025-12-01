from ir import Function, Block, Op, Phi
from typing import Dict, List, Set, Optional
from min_algorithm import SpillReload, is_last_use
import dominators


def is_live_in(block: Block, val_idx: int) -> bool:
    """Check if a variable is live-in to a block."""
    return val_idx in block.live_in


def is_live_out_of_op(
    block: Block, op_idx: int, val_idx: int, function: Function
) -> bool:
    """
    Check if a variable is live-out of an operation.

    A variable is live-out of an operation if it's not at its last use.
    """
    return not is_last_use(val_idx, block, op_idx, function)


def choose_color(
    allocated: Set[int], k: int, last_color: Optional[int] = None
) -> Optional[int]:
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


def repair_argument(
    block: Block,
    op: Op,
    op_idx: int,
    u_val_idx: int,
    parallel_copy: List,
    function: Function,
    k: int,
) -> bool:
    """
    Dummy repair function for argument constraints.

    Always succeeds for now.
    """
    return True


def repair_result(
    block: Block,
    op: Op,
    op_idx: int,
    d_val_idx: int,
    parallel_copy: List,
    function: Function,
    k: int,
) -> bool:
    """
    Dummy repair function for result assignment.

    Always succeeds for now.
    """
    return True


def process_operation(
    block: Block,
    op: Op,
    op_idx: int,
    allocated_variables: Set[int],
    var_to_color: Dict[int, int],
    function: Function,
    k: int,
    last_color: Optional[int],
) -> Optional[int]:
    """
    Process a single operation according to Algorithm 3.

    Args:
        block: The basic block containing the operation
        op: The operation to process
        op_idx: Index of the operation in the block
        allocated_variables: Set of val_idx currently allocated to registers
        var_to_color: Dictionary mapping val_idx to current color (ccolor)
        function: The function containing the block
        k: Number of available registers
        last_color: Last assigned color (for round-robin)

    Returns:
        The last assigned color, or None if no color was assigned
    """
    print(op)
    print()

    parallel_copy = []
    dead = set()

    # Get set of currently allocated register colors
    allocated_colors = {
        var_to_color[v] for v in allocated_variables if v in var_to_color
    }

    # Skip phi operations - phi arguments are considered on incoming edges, not here
    if isinstance(op, Phi):
        # Phi operations are handled separately in the main loop
        return last_color
    # INSERT_YOUR_CODE

    # Check argument constraints and release last used colors
    for u_var in op.uses:
        if u_var not in function.value_indices:
            continue

        u_val_idx = function.value_indices[u_var]

        # Populate use_colors from var_to_color
        if u_val_idx in var_to_color:
            op.use_colors[u_var] = var_to_color[u_val_idx]

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
    allocated_colors = {
        var_to_color[v] for v in allocated_variables if v in var_to_color
    }

    # Assign definitions
    for d_var in op.defs:
        if d_var not in function.value_indices:
            continue

        d_val_idx = function.value_indices[d_var]

        # Choose color
        color = choose_color(allocated_colors, k, last_color)
        if color is None:
            success = repair_result(
                block, op, op_idx, d_val_idx, parallel_copy, function, k
            )
            if not success:
                # Graph coloring would be called here, but we skip for now
                return last_color
            # After repair, try again (recompute allocated_colors)
            allocated_colors = {
                var_to_color[v] for v in allocated_variables if v in var_to_color
            }
            color = choose_color(allocated_colors, k, last_color)
            if color is None:
                return last_color

        var_to_color[d_val_idx] = color
        allocated_variables.add(d_val_idx)
        allocated_colors.add(color)
        last_color = color
        # Store color in instruction's def_colors map
        op.def_colors[d_var] = color
        print(op)

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


def process_spills_reloads(
    position: int,
    spills_reloads: List[SpillReload],
    allocated_variables: Set[int],
    var_to_color: Dict[int, int],
    k: int,
    last_color: Optional[int],
    block: Block,
    function: Function,
) -> Optional[int]:
    """
    Process all spills/reloads at the given position.

    Args:
        position: Instruction position where spills/reloads occur
        spills_reloads: List of all spill/reload operations for the block
        allocated_variables: Set of val_idx currently allocated to registers
        var_to_color: Dictionary mapping val_idx to current color
        k: Number of available registers
        last_color: Last assigned color (for round-robin)
        block: The basic block containing the operations
        function: The function containing the block

    Returns:
        Updated last_color after processing spills/reloads
    """
    # Filter spills/reloads at this position
    for sr in spills_reloads:
        if sr.position != position:
            continue

        # Get set of currently allocated register colors
        allocated_colors = {
            var_to_color[v] for v in allocated_variables if v in var_to_color
        }

        val_idx = sr.val_idx

        if sr.type == "spill":
            # Spill: get current color and remove from allocated set
            if val_idx in var_to_color:
                color = var_to_color[val_idx]
                sr.color = color
                # Remove from allocated set (variable is being spilled)
                allocated_variables.discard(val_idx)
                allocated_colors.discard(color)
        elif sr.type == "reload":
            # Reload: try to use original color if available, otherwise allocate new
            color = None
            # Check if variable had a color before (from original definition)
            if val_idx in var_to_color:
                original_color = var_to_color[val_idx]
                # If original color is available, use it
                if original_color not in allocated_colors:
                    color = original_color

            # If original color not available, allocate a new one
            if color is None:
                color = choose_color(allocated_colors, k, last_color)

            # If still no color (all registers full), try to evict a dead variable
            if color is None:
                # Find a dead variable to evict
                for evict_val_idx in list(allocated_variables):
                    # Check if this variable is dead at the current position (not live-out)
                    instr_idx = position if position < len(block.instructions) else len(block.instructions) - 1
                    if not is_live_out_of_op(block, instr_idx, evict_val_idx, function):
                        # Evict this variable
                        allocated_variables.discard(evict_val_idx)
                        evict_color = var_to_color.pop(evict_val_idx, None)
                        if evict_color is not None:
                            allocated_colors.discard(evict_color)
                        # Now try to allocate color again
                        color = choose_color(allocated_colors, k, last_color)
                        if color is not None:
                            break
            if color is None:
                # Find a spilled variable (already not allocated) with the furthest next-use-distance
                max_dist = -1
                furthest_spilled_val_idx = None
                for v in var_to_color:
                    if v not in allocated_variables:
                        # Get the next-use distance in the current block, from the current position
                        instr_idx = position if position < len(block.instructions) else len(block.instructions) - 1
                        dist = float('inf')
                        if hasattr(block, "next_use_distances_by_val") and v in block.next_use_distances_by_val:
                            # Find minimal next use >= instr_idx
                            found = False
                            for use_pos in block.next_use_distances_by_val[v]:
                                if use_pos >= instr_idx:
                                    dist = use_pos - instr_idx
                                    found = True
                                    break
                            if not found:
                                dist = float('inf')
                        # Prefer largest (furthest) distance, breaking ties by lowest variable index (for determinism)
                        if dist > max_dist or (dist == max_dist and (furthest_spilled_val_idx is None or v < furthest_spilled_val_idx)):
                            max_dist = dist
                            furthest_spilled_val_idx = v
                # If found, reuse its color
                if furthest_spilled_val_idx is not None:
                    color = var_to_color[furthest_spilled_val_idx]
            if color is None:
                raise Exception("No color available for reload")
            if color is not None:
                sr.color = color
                var_to_color[val_idx] = color
                allocated_variables.add(val_idx)
                allocated_colors.add(color)
                last_color = color

    return last_color


def fix_global_color(
    block: Block,
    op_idx: Optional[int],
    allocated_variables: Set[int],
    var_to_color: Dict[int, int],
    function: Function,
    k: int,
):
    """
    Fix global color at the last point of the block where we can insert code.

    This is called when op.next is None or op.next.isLateOperation.
    For now, this is a placeholder - the actual implementation would handle
    late operations and edge splitting.
    """
    # Colors are now stored directly on instructions, so nothing to do here
    pass


def color_program(
    function: Function, k: int, spills_reloads: Dict[str, List[SpillReload]]
) -> None:
    """
    Tree-scan coloring algorithm (Algorithm 2).

    Assigns register colors to SSA values using dominance-based inheritance.
    Colors are stored directly on Op and Phi instructions.
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
    # Per-block state
    block_allocated_variables: Dict[str, Set[int]] = {}
    block_var_to_color: Dict[str, Dict[int, int]] = {}
    block_last_color: Dict[str, Optional[int]] = {}

    # Process blocks in reverse post-order
    for block_name in reverse_postorder:
        block = function.blocks[block_name]

        # Initialize set of occupied registers
        if (
            block_name == entry
            or idom.get(block_name) is None
            or idom.get(block_name) == block_name
        ):
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

        # Get spills/reloads for this block (already sorted by position)
        block_operations = spills_reloads.get(block_name, [])

        # Forward traversal of the operations
        phi_count = 0
        for phi in block.phis():
            phi_count += 1

        # Process phi operations first (they're at the beginning)
        # Note: spills/reloads cannot occur before or between phis
        # Phi arguments are considered on incoming edges, but we still need to assign colors to phi destinations
        for phi_idx, phi in enumerate(block.phis()):
            
            # Get set of currently allocated register colors
            allocated_colors = {
                var_to_color[v] for v in allocated_variables if v in var_to_color
            }

            # Process phi destination
            if phi.dest in function.value_indices:
                d_val_idx = function.value_indices[phi.dest]
                # Choose color for phi destination
                color = choose_color(allocated_colors, k, last_color)
                if color is None:
                    success = repair_result(
                        block, phi, phi_idx, d_val_idx, [], function, k
                    )
                    if not success:
                        # Graph coloring would be called here, but we skip for now
                        pass
                    else:
                        # After repair, try again (recompute allocated_colors)
                        allocated_colors = {
                            var_to_color[v]
                            for v in allocated_variables
                            if v in var_to_color
                        }
                        color = choose_color(allocated_colors, k, last_color)

                if color is not None:
                    var_to_color[d_val_idx] = color
                    allocated_variables.add(d_val_idx)
                    last_color = color
                    # Store color in phi's dest_color field
                    phi.dest_color = color

                    # Release dead definitions
                    if not is_live_out_of_op(block, phi_idx, d_val_idx, function):
                        allocated_variables.discard(d_val_idx)
                        if d_val_idx in var_to_color:
                            allocated_colors.discard(var_to_color[d_val_idx])

        print(phi_count)
        # Process remaining operations (non-phi)
        for op_idx, instr in enumerate(block.instructions[phi_count:], start=phi_count):
            # Process spills/reloads at this instruction position
            last_color = process_spills_reloads(
                op_idx,
                block_operations,
                allocated_variables,
                var_to_color,
                k,
                last_color,
                block,
                function,
            )
            
            if isinstance(instr, Op):
                last_color = process_operation(
                    block,
                    instr,
                    op_idx,
                    allocated_variables,
                    var_to_color,
                    function,
                    k,
                    last_color,
                )
            elif isinstance(instr, Phi):
                # Shouldn't happen after phi_count, but handle it anyway by skipping
                pass

            # Check if this is the last point where we can insert code
            # (op.next is None or op.next.isLateOperation)
            # For now, we check if this is the last instruction
            if op_idx == len(block.instructions) - 1:
                fix_global_color(
                    block, op_idx, allocated_variables, var_to_color, function, k
                )
        # Store state for this block
        block_allocated_variables[block_name] = allocated_variables.copy()
        block_var_to_color[block_name] = var_to_color.copy()
        block_last_color[block_name] = last_color
