# todo: non reducible control flow
import heapq
import math
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from ir import Function, Block, Op, Phi, val_as_phi


@dataclass
class LoopNode:
    """Represents a node in the loop-nesting forest."""

    block_name: str  # Name of the block (or loop header)
    is_loop: bool  # True if this represents a loop header
    children: List["LoopNode"]
    parent: "LoopNode" = None


@dataclass
class LoopAnalysisInfo:
    """Metadata needed to propagate next-use distances inside a loop."""

    header: str
    blocks: Set[str]
    block_lengths: Dict[str, int]
    succs_in_loop: Dict[str, List[str]]
    preds_in_loop: Dict[str, List[str]]
    local_uses: Dict[str, Dict[str, float]]
    best_entry: Dict[str, Dict[str, float]] = None


def compute_predecessors_and_use_def_sets(function: Function) -> None:
    """
    Build predecessor lists and compute USE, DEF, PhiUses, and PhiDefs sets for each block.
    USE set only includes variables used before they are defined in the block.

    Args:
        function: The Function object with blocks
    """
    # Initialize sets for all blocks
    for block in function.blocks.values():
        block.use_set = set()
        block.def_set = set()
        block.phi_uses = set()
        block.phi_defs = set()

    # Single pass over all blocks
    for block_name, block in function.blocks.items():

        # Track variables that have been defined so far in this block
        defined_so_far = set()

        for instr in block.instructions:
            if isinstance(instr, Op):
                # Add uses that haven't been defined yet in this block
                for use in instr.uses:
                    if use not in defined_so_far:
                        block.use_set.add(use)
                # Add defs
                block.def_set.update(instr.defs)
                defined_so_far.update(instr.defs)
            elif isinstance(instr, Phi):
                # Add phi destination to phi_defs
                block.phi_defs.add(instr.dest)
                defined_so_far.add(instr.dest)
                # Add all incoming values to phi_uses
                for incoming in instr.incomings:
                    function.blocks[incoming.block].phi_uses.add(incoming.value)

        # Build predecessors from successors
        for successor in block.successors:
            if successor in function.blocks:
                function.blocks[successor].predecessors.append(block_name)
            else:
                raise ValueError(
                    f"Block '{block_name}' has successor '{successor}' that doesn't exist"
                )


def build_loop_forest(
    function: Function,
) -> Tuple[Dict[str, LoopNode], Set[Tuple[str, str]], Dict[str, Set[str]]]:
    """
    Build loop-nesting forest for reducible graphs and identify loop edges.

    Args:
        function: The Function object with blocks

    Returns:
        Tuple of (loop_forest_dict, loop_edges_set, loop_membership_dict)
        - loop_forest_dict: Maps block names to their LoopNode in the forest
        - loop_edges_set: Set of (source, target) tuples that are loop edges
        - loop_membership_dict: Maps loop headers to sets of blocks in each loop
    """
    # This is a simplified implementation for reducible graphs
    # A full implementation would use Tarjan's algorithm or similar

    # For now, implement a basic DFS-based loop detection
    # This assumes the graph is reducible (as per the problem statement)

    visited = set()
    visiting = set()
    loop_headers = set()
    back_edges = set()
    loop_forest = {}

    def dfs(block_name: str, predecessor: str = None):
        """DFS traversal to find loops and build forest structure."""
        if block_name in visiting:
            # Found a back edge - this indicates a loop
            loop_headers.add(block_name)
            # The back edge is from predecessor to block_name
            if predecessor is not None:
                back_edges.add((predecessor, block_name))
            return

        if block_name in visited:
            return

        visiting.add(block_name)

        for successor in function.blocks[block_name].successors:
            dfs(successor, block_name)

        visiting.remove(block_name)
        visited.add(block_name)

        # Create LoopNode for this block
        loop_forest[block_name] = LoopNode(
            block_name=block_name, is_loop=block_name in loop_headers, children=[]
        )

    # Find a root block (one with no predecessors)
    root_candidates = [
        name for name, block in function.blocks.items() if not block.predecessors
    ]

    if not root_candidates:
        # If no blocks have no predecessors, pick any block as root
        root_candidates = list(function.blocks.keys())[:1]

    # Start DFS from each root candidate
    for root in root_candidates:
        if root not in visited:
            dfs(root)

    # Build parent-child relationships in the forest
    # For this simplified version, we'll create a basic tree structure
    # A full implementation would need proper loop nesting analysis

    # Compute loop membership
    loop_membership = {}
    for loop_header in [
        node.block_name for node in loop_forest.values() if node.is_loop
    ]:
        loop_blocks = {loop_header}
        # Add all sources of back edges to this header
        for src, tgt in back_edges:
            if tgt == loop_header:
                loop_blocks.add(src)
        loop_membership[loop_header] = loop_blocks

    # Build hierarchy based on loop membership
    for loop_header, loop_blocks in loop_membership.items():
        for block in loop_blocks:
            if block != loop_header and block in loop_forest:
                loop_forest[loop_header].children.append(loop_forest[block])
                loop_forest[block].parent = loop_forest[loop_header]

    return loop_forest, back_edges, loop_membership


def expand_loop_blocks(
    header: str, loop_edges: Set[Tuple[str, str]], function: Function
) -> Set[str]:
    """
    Expand the loop membership for a header by walking backwards from all tails.
    """
    members = {header}
    tails = [src for src, tgt in loop_edges if tgt == header]
    stack = list(tails)
    members.update(tails)

    while stack:
        current = stack.pop()
        block = function.blocks.get(current)
        if not block:
            continue
        for pred in block.predecessors:
            if pred == header:
                continue
            if pred not in members and pred in function.blocks:
                members.add(pred)
                stack.append(pred)
    return members


def compute_block_local_uses(
    function: Function, block_name: str, in_loop_successors: List[str]
) -> Dict[str, float]:
    """
    Collect the earliest use distance for each variable within the given block
    as well as phi-uses that occur immediately in successor blocks.
    """
    block = function.blocks[block_name]
    local_uses: Dict[str, float] = {}

    for idx, instr in enumerate(block.instructions):
        if isinstance(instr, Op):
            for use in instr.uses:
                prev = local_uses.get(use, math.inf)
                if idx < prev:
                    local_uses[use] = idx

    block_len = len(block.instructions)
    for succ in in_loop_successors:
        succ_block = function.blocks[succ]
        for instr in succ_block.instructions:
            if isinstance(instr, Phi):
                for incoming in instr.incomings:
                    if incoming.block == block_name:
                        var = incoming.value
                        prev = local_uses.get(var, math.inf)
                        if block_len < prev:
                            local_uses[var] = block_len

    return local_uses


def gather_loop_analysis(
    function: Function,
    loop_forest: Dict[str, LoopNode],
    loop_edges: Set[Tuple[str, str]],
) -> Dict[str, LoopAnalysisInfo]:
    """
    Build metadata for each loop needed to propagate next-use distances.
    """
    loop_infos: Dict[str, LoopAnalysisInfo] = {}

    for block_name, node in loop_forest.items():
        if not node.is_loop or block_name not in function.blocks:
            continue

        blocks = expand_loop_blocks(block_name, loop_edges, function)
        if not blocks:
            continue

        block_lengths = {
            name: len(function.blocks[name].instructions)
            for name in blocks
            if name in function.blocks
        }
        succs_in_loop: Dict[str, List[str]] = {}
        preds_in_loop: Dict[str, List[str]] = {name: [] for name in blocks}

        for name in blocks:
            block = function.blocks.get(name)
            if not block:
                continue
            in_loop_successors = [succ for succ in block.successors if succ in blocks]
            succs_in_loop[name] = in_loop_successors
            for succ in in_loop_successors:
                preds_in_loop.setdefault(succ, []).append(name)

        local_uses = {
            name: compute_block_local_uses(
                function, name, succs_in_loop.get(name, [])
            )
            for name in blocks
            if name in function.blocks
        }

        loop_infos[block_name] = LoopAnalysisInfo(
            header=block_name,
            blocks=blocks,
            block_lengths=block_lengths,
            succs_in_loop=succs_in_loop,
            preds_in_loop=preds_in_loop,
            local_uses=local_uses,
        )

    return loop_infos


def get_loop_processing_order(loop_forest: Dict[str, LoopNode]) -> List[str]:
    """
    Produce a list of loop headers in bottom-up order (children before parents).
    """
    order: List[str] = []
    visited: Set[str] = set()

    def dfs(node: LoopNode) -> None:
        if node.block_name in visited:
            return
        visited.add(node.block_name)
        for child in node.children:
            dfs(child)
        if node.is_loop:
            order.append(node.block_name)

    for node in loop_forest.values():
        if node.parent is None:
            dfs(node)

    return order


def compute_loop_best_entries(
    loop_infos: Dict[str, LoopAnalysisInfo], loop_forest: Dict[str, LoopNode]
) -> None:
    """
    For each loop, compute the minimal distances from every block entry to the
    next use that stays within the same loop body.
    """
    order = get_loop_processing_order(loop_forest)

    for header in order:
        info = loop_infos.get(header)
        if not info:
            continue

        best: Dict[str, Dict[str, float]] = {block: {} for block in info.blocks}
        heap: List[Tuple[float, str, str]] = []

        for block in info.blocks:
            uses = dict(info.local_uses.get(block, {}))
            if block in loop_infos and block != header:
                child_info = loop_infos[block]
                if child_info.best_entry and child_info.best_entry.get(block):
                    for var, dist in child_info.best_entry[block].items():
                        prev = uses.get(var, math.inf)
                        if dist < prev:
                            uses[var] = dist

            for var, dist in uses.items():
                if math.isinf(dist):
                    continue
                if dist < best[block].get(var, math.inf):
                    best[block][var] = dist
                    heapq.heappush(heap, (dist, block, var))

        while heap:
            dist, block, var = heapq.heappop(heap)
            if dist != best[block][var]:
                continue
            for pred in info.preds_in_loop.get(block, []):
                block_len = info.block_lengths.get(pred, 0)
                new_dist = dist + block_len
                if new_dist < best[pred].get(var, math.inf):
                    best[pred][var] = new_dist
                    heapq.heappush(heap, (new_dist, pred, var))

        info.best_entry = best


def apply_loop_best_entries(
    function: Function, loop_infos: Dict[str, LoopAnalysisInfo]
) -> None:
    """
    Push the computed loop distances back into each block's live_in/out sets.
    """
    for info in loop_infos.values():
        if not info.best_entry:
            continue
        # Collect all variables defined within this loop
        loop_def_vars = set()
        for block_name in info.blocks:
            block = function.blocks.get(block_name)
            if block:
                loop_def_vars.update(block.def_set)
                loop_def_vars.update(block.phi_defs)

        for block_name, entry_map in info.best_entry.items():
            block = function.blocks.get(block_name)
            if not block:
                continue
            block_len = info.block_lengths.get(block_name, 0)
            for var, dist in entry_map.items():
                # Don't propagate variables that are defined within the loop
                # to any block within the loop
                if var in loop_def_vars:
                    continue
                if dist < block.live_in.get(var, math.inf):
                    block.live_in[var] = dist
                if dist >= block_len:
                    exit_dist = dist - block_len
                    if exit_dist < block.live_out.get(var, math.inf):
                        block.live_out[var] = exit_dist


def compute_loop_exit_edges(
    loop_membership: Dict[str, Set[str]], function: Function
) -> Set[Tuple[str, str]]:
    """
    Identify loop exit edges.

    An exit edge is an edge (src, dst) where src is in a loop but dst is not in the same loop.

    Args:
        loop_membership: Dictionary mapping loop headers to sets of blocks in each loop
        function: The Function object

    Returns:
        Set of (source, target) tuples representing exit edges
    """
    exit_edges = set()

    # Create reverse mapping: block -> set of loops it belongs to
    block_to_loops = {}
    for loop_header, loop_blocks in loop_membership.items():
        for block in loop_blocks:
            if block not in block_to_loops:
                block_to_loops[block] = set()
            block_to_loops[block].add(loop_header)

    # Check each edge in the CFG
    for src_block_name, src_block in function.blocks.items():
        for dst_block_name in src_block.successors:
            # Check if src is in any loop
            src_loops = block_to_loops.get(src_block_name, set())
            dst_loops = block_to_loops.get(dst_block_name, set())

            # If src is in a loop but dst is not in any of the same loops, it's an exit edge
            if src_loops and not dst_loops.intersection(src_loops):
                exit_edges.add((src_block_name, dst_block_name))

    return exit_edges


def postorder_traversal_reduced_cfg(
    function: Function, loop_edges: Set[Tuple[str, str]]
) -> List[str]:
    """
    Perform postorder traversal of the reduced CFG FL(G) by building adjacency list on-the-fly
    while filtering out loop edges.

    Args:
        function: The Function object
        loop_edges: Set of (source, target) tuples that are loop edges

    Returns:
        List of block names in postorder traversal of FL(G)
    """
    # Build adjacency list on-the-fly while filtering loop edges
    adj = {}
    all_blocks = set(function.blocks.keys())

    for block in all_blocks:
        adj[block] = []

    for block_name, block in function.blocks.items():
        for successor in block.successors:
            edge = (block_name, successor)
            if edge not in loop_edges:
                adj[block_name].append(successor)

    # Perform postorder traversal
    visited = set()
    postorder = []

    def dfs(block_name: str):
        visited.add(block_name)
        for neighbor in adj[block_name]:
            if neighbor not in visited:
                dfs(neighbor)
        postorder.append(block_name)

    # Visit all nodes
    for block in all_blocks:
        if block not in visited:
            dfs(block)

    return postorder


def compute_initial_liveness(
    function: Function,
    loop_forest: Dict[str, LoopNode],
    loop_edges: Set[Tuple[str, str]],
    loop_membership: Dict[str, Set[str]],
) -> None:
    """
    Phase 1: Compute initial liveness sets using postorder traversal of FL(G).

    Args:
        function: The Function object
        loop_forest: Loop forest structure
        loop_edges: Set of loop edges to exclude
        loop_membership: Mapping of loop headers to their member blocks
    """
    # Compute exit edges for distance penalties
    exit_edges = compute_loop_exit_edges(loop_membership, function)

    # Perform postorder traversal of reduced CFG (FL(G))
    postorder = postorder_traversal_reduced_cfg(function, loop_edges)

    # Initialize live sets (already done in Block.__init__)

    # Process blocks in postorder using the helper function
    for block_name in postorder:
        recompute_block_live_sets(function, block_name, exit_edges)


def recompute_block_live_sets(
    function: Function,
    block_name: str,
    exit_edges: Set[Tuple[str, str]],
    restrict_vars: Set[str] = None
) -> None:
    """
    Recompute live_out and live_in for a block based on current successor live_in values.

    Args:
        function: The Function object
        block_name: Name of block to recompute
        exit_edges: Set of loop exit edges for distance penalties
        restrict_vars: If provided, only consider variables in this set for live_out computation
    """
    block = function.blocks[block_name]

    # Compute live_out as the merged live_in from all successors, taking minimums
    # Start with phi_uses (values flowing into phis from this block)
    # Phi uses happen at successor entry, so distance is 0 from block exit
    live_out = {var: 0 for var in block.phi_uses}

    for succ in block.successors:
        if succ in function.blocks:
            succ_block = function.blocks[succ]
            succ_live_in = succ_block.live_in
            for var, val in succ_live_in.items():
                # If restrict_vars is provided, only consider variables in that set
                if restrict_vars is not None and var not in restrict_vars:
                    continue

                adjusted_val = val
                if (block_name, succ) in exit_edges:
                    adjusted_val += 10**9
                # Exclude phi destinations from successor's live_in
                if var in succ_block.phi_defs:
                    # For phi destinations, find the incoming value from this block
                    phi_instr = val_as_phi(function, var)
                    if phi_instr:
                        for incoming in phi_instr.incomings:
                            if incoming.block == block_name:
                                # Include the incoming value instead
                                incoming_val = incoming.value
                                if incoming_val not in live_out:
                                    live_out[incoming_val] = adjusted_val
                                else:
                                    live_out[incoming_val] = min(live_out[incoming_val], adjusted_val)
                    # Don't include the phi destination itself
                else:
                    # Include non-phi variables normally
                    if var not in live_out:
                        live_out[var] = adjusted_val
                    else:
                        live_out[var] = min(live_out[var], adjusted_val)

    block.live_out = live_out

    # LiveIn(B) = (LiveOut(B) - DEF(B)) ∪ USE(B)
    # Start with live_out, excluding def_set and phi_defs keys
    block.live_in = {var: val for var, val in block.live_out.items() if var not in block.def_set and var not in block.phi_defs}
    # Add use_set keys
    for var in block.use_set:
        block.live_in[var] = float("inf")

    # Initialize per-value use position collection
    value_uses: Dict[int, List[int]] = {}

    # Process instructions in reverse to compute actual use distances
    block_len = len(block.instructions)

    # Initialize liveness tracking for pressure calculation
    live_set = set(block.live_out.keys()) if isinstance(block.live_out, dict) else set()
    max_pressure = len(live_set)  # pressure at block exit

    i = block_len
    for instr in reversed(block.instructions):
        i -= 1
        if isinstance(instr, Op):
            defs_set = set(instr.defs)
            uses_set = set(instr.uses)
            for use in instr.uses:
                # Update live_in distance if variable is in live_in
                if use in block.live_in:
                    block.live_in[use] = min(block.live_in[use], i)
                # Collect use positions for per-value analysis (all uses, not just live_in)
                if use in function.value_indices:
                    val_idx = function.value_indices[use]
                    if val_idx not in value_uses:
                        value_uses[val_idx] = []
                    value_uses[val_idx].append(i)

            # Update register pressure: variables live at this point include live_set plus defs and uses
            # live_set currently contains variables that are live after this instruction
            current_live = live_set | defs_set | uses_set
            max_pressure = max(max_pressure, len(current_live))

            # Update live_set for next iteration: remove defs, add uses (backward liveness)
            live_set = (live_set - defs_set) | uses_set
        elif isinstance(instr, Phi):
            defs_set = {instr.dest}
            uses_set = {incoming.value for incoming in instr.incomings}
            for incoming in instr.incomings:
                use = incoming.value
                # Update live_in distance if variable is in live_in
                if use in block.live_in:
                    block.live_in[use] = min(block.live_in[use], i)
                # Collect phi incoming values as uses (all uses, not just live_in)
                if use in function.value_indices:
                    val_idx = function.value_indices[use]
                    if val_idx not in value_uses:
                        value_uses[val_idx] = []
                    value_uses[val_idx].append(i)

            # Update register pressure: variables live at this point include live_set plus defs and uses
            # live_set currently contains variables that are live after this instruction
            current_live = live_set | defs_set | uses_set
            max_pressure = max(max_pressure, len(current_live))

            # Update live_set for next iteration: remove defs, add uses (backward liveness)
            live_set = (live_set - defs_set) | uses_set

    # Final pressure check: pressure at block entry (before any instructions)
    max_pressure = max(max_pressure, len(live_set))

    # Adjust live_in distances for pass-through variables
    for var, dist in block.live_in.items():
        if dist >= block_len and var in block.live_out:
            block.live_in[var] = block.live_out[var] + block_len

    # Convert collected use positions to next_use_distances_by_val
    block.next_use_distances_by_val = {}

    # Create reverse lookup from val_idx to variable name for checking block.live_out
    val_idx_to_var = {idx: var for var, idx in function.value_indices.items()}

    # Process value_uses (now keyed by val_idx) and store in block.next_use_distances_by_val
    for val_idx, use_positions in value_uses.items():
        # Sort to get chronological order (since we collected in reverse)
        sorted_positions = sorted(use_positions)
        # Add liveout distance as the last entry (using final live_out values after propagation)
        var = val_idx_to_var[val_idx]  # Get variable name for live_out check
        if isinstance(block.live_out, dict) and var in block.live_out:
            # live_out[var] is distance from block exit, so add block_len to get distance from start
            sorted_positions.append(block_len + block.live_out[var])
        else:
            # Not live out, append infinity as the last entry
            sorted_positions.append(math.inf)
        block.next_use_distances_by_val[val_idx] = sorted_positions

    # Handle variables that are in live_out but didn't appear in value_uses
    if isinstance(block.live_out, dict):
        for var in block.live_out:
            if var in function.value_indices:
                val_idx = function.value_indices[var]
                # Only add if not already processed above
                if val_idx not in block.next_use_distances_by_val:
                    # Variable is live out but has no uses in this block
                    # Add liveout distance as the only entry
                    block.next_use_distances_by_val[val_idx] = [block_len + block.live_out[var]]

    block.max_register_pressure = max_pressure


def propagate_loop_liveness_and_distances(
    function: Function,
    loop_forest: Dict[str, LoopNode],
    loop_edges: Set[Tuple[str, str]],
    loop_membership: Dict[str, Set[str]]
) -> None:
    """
    Combined phase: Propagate liveness within loop bodies and compute next-use distances.

    Args:
        function: The Function object
        loop_forest: Loop forest structure
        loop_edges: Set of loop edges
        loop_membership: Mapping of loop headers to their member blocks
    """

    # Track which blocks need distance recomputation
    affected_blocks = set()

    # Compute exit edges once for both liveness and distance propagation
    exit_edges = compute_loop_exit_edges(loop_membership, function)

    def loop_tree_dfs(node: LoopNode) -> None:
        """Recursive DFS traversal of the loop forest (Algorithm 3)."""
        if node.is_loop:
            # Collect all live variable NAMES in this loop (not distances)
            loop_live_vars = set()

            # Add live vars from the loop header
            block_n = function.blocks[node.block_name]
            loop_live_vars.update(block_n.live_in.keys())
            loop_live_vars.update(block_n.live_out.keys())

            # Add live vars from all children (recursive)
            def collect_live_vars(n: LoopNode) -> None:
                block = function.blocks[n.block_name]
                loop_live_vars.update(block.live_in.keys())
                loop_live_vars.update(block.live_out.keys())
                for child in n.children:
                    collect_live_vars(child)

            for child in node.children:
                collect_live_vars(child)

            # Propagate all loop live vars to the header and children (with placeholder distances)
            for var in loop_live_vars:
                # Don't add variables to live_in if they're defined in this block
                if var not in block_n.def_set and var not in block_n.phi_defs:
                    block_n.live_in[var] = block_n.live_in.get(var, float('inf'))
                block_n.live_out[var] = block_n.live_out.get(var, float('inf'))

            for child in node.children:
                block_m = function.blocks[child.block_name]
                for var in loop_live_vars:
                    # Don't add variables to live_in if they're defined in this block
                    if var not in block_m.def_set and var not in block_m.phi_defs:
                        block_m.live_in[var] = block_m.live_in.get(var, float('inf'))
                    block_m.live_out[var] = block_m.live_out.get(var, float('inf'))

                # Mark child block for recomputation
                affected_blocks.add(child.block_name)
                # Recursively process child
                loop_tree_dfs(child)

            # Mark header block for recomputation
            affected_blocks.add(node.block_name)

    # Start from root nodes (nodes with no parent)
    roots = [node for node in loop_forest.values() if node.parent is None]
    for root in roots:
        loop_tree_dfs(root)

    # Compute loop-aware next-use distances
    if loop_membership:
        loop_infos = gather_loop_analysis(function, loop_forest, loop_edges)
        if loop_infos:
            compute_loop_best_entries(loop_infos, loop_forest)
            apply_loop_best_entries(function, loop_infos)

            # Add blocks affected by distance computation
            for info in loop_infos.values():
                affected_blocks.update(info.blocks)

    # Recompute distances for all affected blocks in postorder
    if affected_blocks:
        postorder = postorder_traversal_reduced_cfg(function, loop_edges)
        for block_name in postorder:
            if block_name in affected_blocks:
                recompute_block_live_sets(function, block_name, exit_edges)






def check_liveness_correctness(function: Function) -> bool:
    """
    Check correctness of liveness analysis by verifying that every variable in LiveOut(B)
    appears in LiveIn(S) for at least one successor S of B.

    Args:
        function: The Function object to check

    Returns:
        bool: True if liveness is correct, False otherwise

    Raises:
        AssertionError: If liveness correctness check fails with details
    """
    errors = []

    for block_name, block in function.blocks.items():
        # Skip terminal blocks (blocks with no successors) as they don't need to propagate liveness
        if not block.successors:
            continue

        for var in block.live_out:
            found_in_successor = False
            successor_live_ins = []

            for successor in block.successors:
                if successor in function.blocks:
                    succ_block = function.blocks[successor]
                    successor_live_ins.append(
                        f"{successor}: {sorted(succ_block.live_in)}"
                    )
                    if var in succ_block.live_in:
                        found_in_successor = True
                        break
                    # Also check if var flows into a phi in this successor from this block
                    for instr in succ_block.instructions:
                        if isinstance(instr, Phi):
                            for incoming in instr.incomings:
                                if incoming.block == block_name and incoming.value == var:
                                    found_in_successor = True
                                    break
                            if found_in_successor:
                                break
                    if found_in_successor:
                        break

            if not found_in_successor:
                error_msg = (
                    f"Variable '{var}' in LiveOut of block '{block_name}' "
                    f"not found in LiveIn of any successor.\n"
                    f"  LiveOut({block_name}): {sorted(block.live_out)}\n"
                    f"  Successors LiveIn: {successor_live_ins}"
                )
                errors.append(error_msg)

    if errors:
        error_details = "\n\n".join(errors)
        raise AssertionError(f"Liveness correctness check failed:\n\n{error_details}")

    return True



def get_next_use_distance(block: Block, var: str, current_idx: int, function: Function) -> float:
    """
    Get the next-use distance for a variable at a given instruction index.
    
    Args:
        block: The block containing the variable
        var: The variable name
        current_idx: The current instruction index (0-based from block start)
        function: The function containing value_indices mapping
        
    Returns:
        Distance to next use, or math.inf if no future use exists.
        Returns 0 if the variable is defined at the current instruction.
    """
    # Check if variable is defined at the current instruction
    if current_idx < len(block.instructions):
        current_instr = block.instructions[current_idx]
        if isinstance(current_instr, Op) and var in current_instr.defs:
            return 0.0
        elif isinstance(current_instr, Phi) and current_instr.dest == var:
            return 0.0
    
    if var not in function.value_indices:
        return math.inf
    val_idx = function.value_indices[var]
    if not hasattr(block, 'next_use_distances_by_val') or val_idx not in block.next_use_distances_by_val:
        return math.inf
    use_positions = block.next_use_distances_by_val[val_idx]
    for pos in use_positions:
        if pos >= current_idx:
            return pos - current_idx
    return math.inf




def compute_liveness(function: Function) -> None:
    """
    Main function that orchestrates the two-phase liveness analysis.

    Args:
        function: The Function object to analyze
    """
    # Phase 0: Setup
    compute_predecessors_and_use_def_sets(function)

    # Build loop forest and identify loop edges
    loop_forest, loop_edges, loop_membership = build_loop_forest(function)

    # Phase 1: Initial liveness computation
    compute_initial_liveness(function, loop_forest, loop_edges, loop_membership)

    # Phase 2: Loop propagation and distance computation
    propagate_loop_liveness_and_distances(function, loop_forest, loop_edges, loop_membership)


