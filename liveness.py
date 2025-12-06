# todo: non reducible control flow
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from ir import Block, Function, Op, Phi, val_as_phi


@dataclass
class LoopNode:
    """Represents a node in the loop-nesting forest."""

    block_idx: int  # RPO index of the block (or loop header)
    is_loop: bool  # True if this represents a loop header
    children: List["LoopNode"]
    parent: "LoopNode" = None


def compute_predecessors_and_use_def_sets(function: Function) -> None:
    """
    Build predecessor lists and compute USE, DEF, PhiUses, and PhiDefs sets for each block.
    USE set only includes variables used before they are defined in the block.

    Args:
        function: The Function object with blocks
    """

    # Single pass over all blocks
    for block_name, block in function.blocks.items():
        # Track val_idx that have been defined so far in this block
        defined_so_far = set()

        # Process phi instructions first (they're always at the beginning)
        phi_count = 0
        for phi in block.phis():
            # Add phi destination to phi_defs
            if phi.dest in function.value_indices:
                dest_idx = function.value_indices[phi.dest]
                block.phi_defs.add(dest_idx)
                defined_so_far.add(dest_idx)
            # Add all incoming values to phi_uses
            for incoming in phi.incomings:
                if incoming.value in function.value_indices:
                    val_idx = function.value_indices[incoming.value]
                    function.blocks[incoming.block].phi_uses.add(val_idx)
            phi_count += 1

        # Process remaining instructions (skip phis since we already processed them)
        for instr in block.instructions[phi_count:]:
            if isinstance(instr, Op):
                # Add uses that haven't been defined yet in this block
                for use in instr.uses:
                    if use in function.value_indices:
                        use_idx = function.value_indices[use]
                        if use_idx not in defined_so_far:
                            block.use_set.add(use_idx)
                # Add defs
                for def_var in instr.defs:
                    if def_var in function.value_indices:
                        def_idx = function.value_indices[def_var]
                        block.def_set.add(def_idx)
                        defined_so_far.add(def_idx)

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
) -> Tuple[
    Dict[str, LoopNode],
    Set[Tuple[str, str]],
    Dict[str, Set[str]],
    Set[Tuple[str, str]],
    List[str],
    List[str],
    Dict[str, int],
]:
    """
    Build loop-nesting forest using the one-pass DFS tagging algorithm of
    Wei et al. (“A New Algorithm for Identifying Loops in Decompilation”).

    Returns:
        Tuple of (loop_forest_dict, loop_edges_set, loop_membership_dict, exit_edges_set, postorder, rpo, name_to_rpo_idx)
        - loop_forest_dict: Maps block names to their LoopNode in the forest
        - loop_edges_set: Set of (source, target) tuples that are loop/back edges
        - loop_membership_dict: Maps loop headers to sets of blocks in each loop
        - exit_edges_set: Set of (source, target) tuples that are loop exit edges
        - postorder: List of block names in postorder traversal
        - rpo: List of block names in reverse postorder (RPO) traversal
        - name_to_rpo_idx: Dictionary mapping block names to their RPO indices
    """

    # Per-block transient metadata
    traversed: Set[str] = set()
    dfsp_pos: Dict[str, int] = {name: 0 for name in function.blocks}
    iloop_header: Dict[str, str] = {name: None for name in function.blocks}
    is_loop_header: Set[str] = set()
    irreducible_headers: Set[str] = set()
    reentry_edges: Set[Tuple[str, str]] = set()
    loop_edges: Set[Tuple[str, str]] = set()
    postorder: List[str] = []

    def tag_lhead(b: str, h: str) -> None:
        """Weave header h into the loop-header list of b according to DFSP positions."""
        if h is None or b == h:
            return

        cur1, cur2 = b, h
        while iloop_header[cur1] is not None:
            ih = iloop_header[cur1]
            if ih == cur2:
                return
            if dfsp_pos[ih] < dfsp_pos[cur2]:
                iloop_header[cur1] = cur2
                cur1, cur2 = cur2, ih
            else:
                cur1 = ih
        iloop_header[cur1] = cur2

    def trav_loops_dfs(b0: str, pos: int) -> str:
        """Single-pass DFS that tags loop headers on demand."""
        traversed.add(b0)
        dfsp_pos[b0] = pos

        for succ in function.blocks[b0].successors:
            if succ not in function.blocks:
                continue

            if succ not in traversed:
                # Case (A): new node
                nh = trav_loops_dfs(succ, pos + 1)
                tag_lhead(b0, nh)
            else:
                if dfsp_pos[succ] > 0:
                    # Case (B): back edge to DFSP
                    is_loop_header.add(succ)
                    loop_edges.add((b0, succ))
                    tag_lhead(b0, succ)
                elif iloop_header[succ] is None:
                    # Case (C): traversed, not on path, not in a loop body
                    continue
                else:
                    h = iloop_header[succ]
                    if h is None:
                        continue
                    if dfsp_pos[h] > 0:
                        # Case (D): successor in loop; header on current path
                        tag_lhead(b0, h)
                    else:
                        # Case (E): re-entry into loop not on current path
                        reentry_edges.add((b0, succ))
                        irreducible_headers.add(h)
                        while iloop_header[h] is not None:
                            h = iloop_header[h]
                            if dfsp_pos[h] > 0:
                                tag_lhead(b0, h)
                                break
                            irreducible_headers.add(h)

        postorder.append(b0)
        dfsp_pos[b0] = 0  # clear DFSP position
        return iloop_header[b0]

    # Pick entry roots: blocks without predecessors, or fall back to an arbitrary block
    roots = [name for name, block in function.blocks.items() if not block.predecessors]
    if not roots and function.blocks:
        roots = [next(iter(function.blocks))]

    # Traverse all reachable components
    for root in roots:
        if root not in traversed:
            trav_loops_dfs(root, 1)

    # Visit any remaining disconnected blocks to ensure coverage
    for name in function.blocks:
        if name not in traversed:
            trav_loops_dfs(name, 1)

    # RPO and index mapping
    rpo = list(reversed(postorder))
    name_to_rpo_idx = {block_name: idx for idx, block_name in enumerate(rpo)}

    # Create LoopNode instances
    loop_forest: Dict[str, LoopNode] = {}
    for block_name in rpo:
        block_idx = name_to_rpo_idx[block_name]
        loop_forest[block_name] = LoopNode(
            block_idx=block_idx,
            is_loop=block_name in is_loop_header,
            children=[],
        )
        # Optional irreducible marker on headers
        if block_name in irreducible_headers:
            setattr(loop_forest[block_name], "irreducible", True)

    # Compute loop membership from tagged headers (include nested headers)
    loop_membership: Dict[str, Set[str]] = {}
    for block_name in function.blocks:
        header = iloop_header[block_name]
        seen: Set[str] = set()
        while header is not None and header not in seen:
            loop_membership.setdefault(header, set()).add(block_name)
            seen.add(header)
            header = iloop_header.get(header)
    for header in is_loop_header:
        loop_membership.setdefault(header, set()).add(header)

    # Build parent-child relationships based on immediate innermost header
    for block_name in function.blocks:
        header = iloop_header[block_name]
        if header and header in loop_forest and block_name in loop_forest:
            loop_forest[header].children.append(loop_forest[block_name])
            loop_forest[block_name].parent = loop_forest[header]

    # Reverse mapping: block -> loops containing it
    block_to_loops: Dict[str, Set[str]] = {}
    for header, blocks in loop_membership.items():
        for block in blocks:
            block_to_loops.setdefault(block, set()).add(header)

    # Loop depth and exit edges
    exit_edges: Set[Tuple[str, str]] = set()
    for block_name, block in function.blocks.items():
        block.loop_depth = len(block_to_loops.get(block_name, set()))

        for succ in block.successors:
            src_loops = block_to_loops.get(block_name, set())
            dst_loops = block_to_loops.get(succ, set())
            if src_loops and not src_loops.intersection(dst_loops):
                exit_edges.add((block_name, succ))

    return loop_forest, loop_edges, loop_membership, exit_edges, postorder, rpo, name_to_rpo_idx


def get_loop_processing_order(loop_forest: Dict[str, LoopNode], rpo: List[str]) -> List[str]:
    """
    Produce a list of loop headers in bottom-up order (children before parents).
    
    Args:
        loop_forest: Dictionary mapping block names to LoopNodes
        rpo: List of block names in reverse postorder (for index-to-name lookup)
    
    Returns:
        List of loop header block names in bottom-up order
    """
    order: List[str] = []
    visited: Set[int] = set()

    def dfs(node: LoopNode) -> None:
        if node.block_idx in visited:
            return
        visited.add(node.block_idx)
        for child in node.children:
            dfs(child)
        if node.is_loop:
            # Look up block name from RPO index
            if 0 <= node.block_idx < len(rpo):
                order.append(rpo[node.block_idx])

    for node in loop_forest.values():
        if node.parent is None:
            dfs(node)

    return order


def compute_initial_liveness(
    function: Function,
    loop_forest: Dict[str, LoopNode],
    loop_edges: Set[Tuple[str, str]],
    loop_membership: Dict[str, Set[str]],
    exit_edges: Set[Tuple[str, str]],
    postorder: List[str],
) -> None:
    """
    Phase 1: Compute initial liveness sets using postorder traversal of FL(G).

    Args:
        function: The Function object
        loop_forest: Loop forest structure
        loop_edges: Set of loop edges to exclude
        loop_membership: Mapping of loop headers to their member blocks
        exit_edges: Set of loop exit edges for distance penalties
        postorder: Postorder traversal of the reduced CFG FL(G)
    """

    # Process blocks in postorder using the helper function
    for block_name in postorder:
        recompute_block_live_sets(function, block_name, set())


def recompute_block_live_sets(
    function: Function,
    block_name: str,
    exit_edges: Set[Tuple[str, str]],
    restrict_vars: Set[int] = None,
) -> None:
    """
    Recompute live_out and live_in for a block based on current successor live_in values.

    Args:
        function: The Function object
        block_name: Name of block to recompute
        exit_edges: Set of loop exit edges for distance penalties
        restrict_vars: If provided, only consider val_idx in this set for live_out computation
    """
    block = function.blocks[block_name]

    # Compute live_out as the merged live_in from all successors, taking minimums
    # Start with phi_uses (values flowing into phis from this block)
    # Phi uses happen at successor entry, so distance is 0 from block exit
    live_out: Dict[int, float] = {val_idx: 0 for val_idx in block.phi_uses}

    for succ in block.successors:
        succ_block = function.blocks[succ]
        succ_live_in = succ_block.live_in
        for val_idx, val in succ_live_in.items():
            # If restrict_vars is provided, only consider variables in that set
            if restrict_vars is not None and val_idx not in restrict_vars:
                continue

            adjusted_val = val
            if (block_name, succ) in exit_edges:
                adjusted_val += 10**9
            # Exclude phi destinations from successor's live_in
            if val_idx in succ_block.phi_defs:
                # For phi destinations, find the incoming value from this block
                phi_instr = val_as_phi(function, val_idx)
                if phi_instr:
                    incoming_val = phi_instr.incoming_val_for_block(block_name)
                    if incoming_val is not None and incoming_val in function.value_indices:
                        incoming_val_idx = function.value_indices[incoming_val]
                        if incoming_val_idx not in live_out:
                            live_out[incoming_val_idx] = adjusted_val
                        else:
                            live_out[incoming_val_idx] = min(
                                live_out[incoming_val_idx], adjusted_val
                            )
                # Don't include the phi destination itself
            else:
                # Include non-phi variables normally
                if val_idx not in live_out:
                    live_out[val_idx] = adjusted_val
                else:
                    live_out[val_idx] = min(live_out[val_idx], adjusted_val)

    block.live_out = live_out

    # LiveIn(B) = (LiveOut(B) - DEF(B)) ∪ USE(B)
    # Start with live_out, excluding def_set and phi_defs keys
    block.live_in = {
        val_idx: val
        for val_idx, val in block.live_out.items()
        if val_idx not in block.def_set and val_idx not in block.phi_defs
    }
    # Add use_set keys
    for val_idx in block.use_set:
        block.live_in[val_idx] = float("inf")

    # Initialize per-value use position collection
    block.next_use_distances_by_val = {}

    # Process instructions in reverse to compute actual use distances
    block_len = len(block.instructions)

    # Initialize liveness tracking for pressure calculation
    live_set = set(block.live_out.keys()) if isinstance(block.live_out, dict) else set()
    max_pressure = len(live_set)  # pressure at block exit

    i = block_len
    for instr in reversed(block.instructions):
        i -= 1
        if isinstance(instr, Op):
            # Convert defs and uses to val_idx sets for pressure calculation
            defs_set = set()
            uses_set = set()
            for def_var in instr.defs:
                if def_var in function.value_indices:
                    defs_set.add(function.value_indices[def_var])
            for use_var in instr.uses:
                if use_var in function.value_indices:
                    use_idx = function.value_indices[use_var]
                    uses_set.add(use_idx)
                    # Update live_in distance if variable is in live_in
                    if use_idx in block.live_in:
                        block.live_in[use_idx] = min(block.live_in[use_idx], i)
                    # Collect use positions for per-value analysis (all uses, not just live_in)
                    if use_idx not in block.next_use_distances_by_val:
                        block.next_use_distances_by_val[use_idx] = []
                    block.next_use_distances_by_val[use_idx].append(i)

            # Update register pressure: variables live at this point include live_set plus defs and uses
            # live_set currently contains variables that are live after this instruction
            current_live = live_set | defs_set | uses_set
            max_pressure = max(max_pressure, len(current_live))

            # Update live_set for next iteration: remove defs, add uses (backward liveness)
            live_set = (live_set - defs_set) | uses_set
        elif isinstance(instr, Phi):
            # Convert phi dest and incomings to val_idx sets
            defs_set = set()
            uses_set = set()
            if instr.dest in function.value_indices:
                defs_set.add(function.value_indices[instr.dest])
            for incoming in instr.incomings:
                if incoming.value in function.value_indices:
                    use_idx = function.value_indices[incoming.value]
                    uses_set.add(use_idx)
                    # Update live_in distance if variable is in live_in
                    if use_idx in block.live_in:
                        block.live_in[use_idx] = min(block.live_in[use_idx], i)
                    # Collect phi incoming values as uses (all uses, not just live_in)
                    if use_idx not in block.next_use_distances_by_val:
                        block.next_use_distances_by_val[use_idx] = []
                    block.next_use_distances_by_val[use_idx].append(i)

            # Update register pressure: variables live at this point include live_set plus defs and uses
            # live_set currently contains variables that are live after this instruction
            current_live = live_set | defs_set | uses_set
            max_pressure = max(max_pressure, len(current_live))

            # Update live_set for next iteration: remove defs, add uses (backward liveness)
            live_set = (live_set - defs_set) | uses_set

    # Final pressure check: pressure at block entry (before any instructions)
    max_pressure = max(max_pressure, len(live_set))

    # Adjust live_in distances for pass-through variables
    for val_idx, dist in block.live_in.items():
        if dist >= block_len and val_idx in block.live_out:
            block.live_in[val_idx] = block.live_out[val_idx] + block_len

    # Finalize next_use_distances_by_val: sort and append live_out distances
    for val_idx, use_positions in block.next_use_distances_by_val.items():
        # Sort to get chronological order (since we collected in reverse)
        use_positions.sort()
        # Add liveout distance as the last entry (using final live_out values after propagation)
        if isinstance(block.live_out, dict) and val_idx in block.live_out:
            # live_out[val_idx] is distance from block exit, so add block_len to get distance from start
            use_positions.append(block_len + block.live_out[val_idx])
        else:
            # Not live out, append infinity as the last entry
            use_positions.append(math.inf)

    # Handle variables that are in live_out but didn't appear in next_use_distances_by_val
    if isinstance(block.live_out, dict):
        for val_idx in block.live_out:
            # Only add if not already processed above
            if val_idx not in block.next_use_distances_by_val:
                # Variable is live out but has no uses in this block
                # Add liveout distance as the only entry
                block.next_use_distances_by_val[val_idx] = [
                    block_len + block.live_out[val_idx]
                ]

    block.max_register_pressure = max_pressure


def propagate_loop_liveness_and_distances(
    function: Function,
    loop_forest: Dict[str, LoopNode],
    loop_edges: Set[Tuple[str, str]],
    loop_membership: Dict[str, Set[str]],
    exit_edges: Set[Tuple[str, str]],
    postorder: List[str],
    rpo: List[str],
) -> None:
    """
    Combined phase: Propagate liveness within loop bodies and compute next-use distances.

    Args:
        function: The Function object
        loop_forest: Loop forest structure
        loop_edges: Set of loop edges
        loop_membership: Mapping of loop headers to their member blocks
        exit_edges: Set of loop exit edges for distance penalties
        postorder: Postorder traversal of the reduced CFG FL(G)
        rpo: List of block names in reverse postorder (for index-to-name lookup)
    """

    # Track which blocks need distance recomputation
    affected_blocks = set()

    def loop_tree_dfs(node: LoopNode) -> None:
        """Recursive DFS traversal of the loop forest (Algorithm 3)."""
        if node.is_loop:
            # Collect all live val_idx in this loop (not distances)
            loop_live_vars = set()

            # Look up block name from RPO index
            block_name = rpo[node.block_idx] if 0 <= node.block_idx < len(rpo) else None
            if block_name is None:
                return

            # Add live vars from the loop header
            block_n = function.blocks[block_name]
            loop_live_vars.update(block_n.live_in.keys())
            loop_live_vars.update(block_n.live_out.keys())

            # Add live vars from all children (recursive)
            def collect_live_vars(n: LoopNode) -> None:
                child_block_name = rpo[n.block_idx] if 0 <= n.block_idx < len(rpo) else None
                if child_block_name is None:
                    return
                block = function.blocks[child_block_name]
                loop_live_vars.update(block.live_in.keys())
                loop_live_vars.update(block.live_out.keys())
                for child in n.children:
                    collect_live_vars(child)

            for child in node.children:
                collect_live_vars(child)

            # Propagate all loop live vars to the header and children (with placeholder distances)
            for val_idx in loop_live_vars:
                # Don't add variables to live_in if they're defined in this block
                if val_idx not in block_n.def_set and val_idx not in block_n.phi_defs:
                    block_n.live_in[val_idx] = block_n.live_in.get(
                        val_idx, float("inf")
                    )
                block_n.live_out[val_idx] = block_n.live_out.get(val_idx, float("inf"))

            for child in node.children:
                child_block_name = rpo[child.block_idx] if 0 <= child.block_idx < len(rpo) else None
                if child_block_name is None:
                    continue
                block_m = function.blocks[child_block_name]
                for val_idx in loop_live_vars:
                    # Don't add variables to live_in if they're defined in this block
                    if (
                        val_idx not in block_m.def_set
                        and val_idx not in block_m.phi_defs
                    ):
                        block_m.live_in[val_idx] = block_m.live_in.get(
                            val_idx, float("inf")
                        )
                    block_m.live_out[val_idx] = block_m.live_out.get(
                        val_idx, float("inf")
                    )

                # Mark child block for recomputation
                affected_blocks.add(child_block_name)
                # Recursively process child
                loop_tree_dfs(child)

            # Mark header block for recomputation
            affected_blocks.add(block_name)

    # Start from root nodes (nodes with no parent)
    roots = [node for node in loop_forest.values() if node.parent is None]
    for root in roots:
        loop_tree_dfs(root)

    # Compute loop-aware next-use distances using BFS
    if loop_membership:
        # Process loops in bottom-up order (inner loops first)
        loop_order = get_loop_processing_order(loop_forest, rpo)

        for header in loop_order:
            blocks = loop_membership.get(header, set())
            if not blocks or header not in function.blocks:
                continue

            header_block = function.blocks[header]
            # Collect all live variables in this loop
            loop_live_vars = set(header_block.live_in.keys())
            loop_live_vars.update(header_block.live_out.keys())

            # Also collect from nested loops
            for child_node in loop_forest[header].children:
                if child_node.is_loop:
                    child_block_name = rpo[child_node.block_idx] if 0 <= child_node.block_idx < len(rpo) else None
                    if child_block_name is None:
                        continue
                    child_block = function.blocks[child_block_name]
                    loop_live_vars.update(child_block.live_in.keys())
                    loop_live_vars.update(child_block.live_out.keys())

            # BFS from header to find minimum distance to first use of each live variable
            min_dist: Dict[int, float] = {}
            visited = set()
            queue = deque([(header, 0)])  # (block_name, distance_from_header_entry)

            while queue:
                block_name, dist = queue.popleft()
                if (
                    block_name in visited
                    or block_name not in blocks
                    or block_name not in function.blocks
                ):
                    continue
                visited.add(block_name)

                block = function.blocks[block_name]
                block_len = len(block.instructions)

                # Check uses in this block's instructions
                for idx, instr in enumerate(block.instructions):
                    if isinstance(instr, Op):
                        for use in instr.uses:
                            if use in function.value_indices:
                                val_idx = function.value_indices[use]
                                if val_idx in loop_live_vars:
                                    use_dist = dist + idx
                                    if (
                                        val_idx not in min_dist
                                        or use_dist < min_dist[val_idx]
                                    ):
                                        min_dist[val_idx] = use_dist

                # Check phi uses in successor blocks (within loop)
                for succ in block.successors:
                    if succ in blocks and succ in function.blocks:
                        succ_block = function.blocks[succ]
                        for phi in succ_block.phis():
                            for incoming in phi.incomings:
                                if incoming.block == block_name:
                                    var = incoming.value
                                    if var in function.value_indices:
                                        val_idx = function.value_indices[var]
                                        if val_idx in loop_live_vars:
                                            # Phi use is at distance block_len from block entry
                                            use_dist = dist + block_len
                                            if (
                                                val_idx not in min_dist
                                                or use_dist < min_dist[val_idx]
                                            ):
                                                min_dist[val_idx] = use_dist

                # Add successors within loop to queue
                for succ in block.successors:
                    if succ in blocks and succ not in visited:
                        queue.append((succ, dist + block_len))

            # Update header's live_in with computed minimum distances
            # Only update variables that are not defined in the header
            for val_idx, d in min_dist.items():
                if (
                    val_idx not in header_block.def_set
                    and val_idx not in header_block.phi_defs
                ):
                    if d < header_block.live_in.get(val_idx, math.inf):
                        header_block.live_in[val_idx] = d

            # Mark all loop blocks for recomputation
            # affected_blocks.update(blocks)

    # Recompute distances for all affected blocks in postorder
    if affected_blocks:
        for block_name in postorder:
            if block_name in affected_blocks:
                recompute_block_live_sets(function, block_name, exit_edges)


def get_next_use_distance(
    block: Block, val_idx: int, current_idx: int, function: Function
) -> float:
    """
    Get the next-use distance for a variable at a given instruction index.

    Args:
        block: The block containing the variable
        val_idx: The value index (integer identifier)
        current_idx: The current instruction index (0-based from block start)
        function: The function containing value_indices mapping

    Returns:
        Distance to next use, or math.inf if no future use exists.
        Returns 0 if the variable is defined at the current instruction.
    """
    # Check if variable is defined at the current instruction
    if current_idx < len(block.instructions):
        current_instr = block.instructions[current_idx]
        if isinstance(current_instr, Op):
            for def_var in current_instr.defs:
                if (
                    def_var in function.value_indices
                    and function.value_indices[def_var] == val_idx
                ):
                    return 0.0
        elif isinstance(current_instr, Phi):
            if (
                current_instr.dest in function.value_indices
                and function.value_indices[current_instr.dest] == val_idx
            ):
                return 0.0

    if (
        not hasattr(block, "next_use_distances_by_val")
        or val_idx not in block.next_use_distances_by_val
    ):
        return math.inf
    use_positions = block.next_use_distances_by_val[val_idx]
    for pos in use_positions:
        if pos >= current_idx:
            return pos - current_idx
    return math.inf


def compute_liveness(function: Function) -> Dict[str, Set[str]]:
    """
    Main function that orchestrates the two-phase liveness analysis.

    Args:
        function: The Function object to analyze

    Returns:
        Dictionary mapping loop headers to sets of blocks in each loop (loop_membership)
    """
    # Phase 0: Setup
    compute_predecessors_and_use_def_sets(function)

    # Build loop forest and identify loop edges
    loop_forest, loop_edges, loop_membership, exit_edges, postorder, rpo, name_to_rpo_idx = build_loop_forest(
        function
    )

    # Phase 1: Initial liveness computation
    compute_initial_liveness(
        function, loop_forest, loop_edges, loop_membership, exit_edges, postorder
    )

    # Phase 2: Loop propagation and distance computation
    propagate_loop_liveness_and_distances(
        function, loop_forest, loop_edges, loop_membership, exit_edges, postorder, rpo
    )

    return loop_membership
