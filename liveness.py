# todo: non reducible control flow
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ir import Block, Function, Op, Phi, val_as_phi


@dataclass
class LoopNode:
    """Represents a node in the loop-nesting forest."""

    block_idx: int  # RPO index of the block (or loop header)
    is_loop: bool  # True if this represents a loop header
    children: List["LoopNode"]
    parent: "LoopNode" = None


@dataclass
class LoopInfo:
    """List-based loop descriptor (analogue of the C++ small-vector layout)."""

    header: str
    level: int
    parent: Optional[int]
    begin: int  # inclusive block index in block_layout
    end: int  # exclusive block index in block_layout
    num_blocks: int
    definitions: int = 0
    definitions_in_childs: int = 0
    irreducible: bool = False
    reentries: List[Tuple[str, str]] = None


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
    List[str],  # block_layout (RPO)
    List[Optional[int]],  # block_loop_map: loop index per block index
    List[LoopInfo],  # loops
    Set[Tuple[str, str]],  # loop/back edges
    Set[Tuple[str, str]],  # exit edges
    List[str],  # postorder
    List[str],  # rpo
    Dict[str, int],  # name_to_rpo_idx
]:
    """
    Build loop descriptors using the one-pass DFS tagging algorithm of
    Wei et al. (“A New Algorithm for Identifying Loops in Decompilation”)
    and emit list-based structures analogous to the C++ layout.

    Returns (block_layout, block_loop_map, loops, loop_edges, exit_edges, postorder, rpo, name_to_rpo_idx)
        - block_layout: List of block names in reverse postorder (RPO) traversal
        - block_loop_map: List aligned with block_layout; entry is the loop index (or None) that owns the block (innermost)
        - loops: List of LoopInfo with header, level, parent index, begin/end range over block_layout, and metadata
        - loop_edges: Set of (source, target) tuples that are loop/back edges
        - exit_edges: Set of (source, target) tuples that are loop exit edges
        - postorder: List of block names in postorder traversal
        - rpo: List of block names in reverse postorder (RPO) traversal
        - name_to_rpo_idx: Dictionary mapping block names to their RPO indices
    """

    # Per-block transient metadata
    traversed: Set[str] = set()
    dfsp_pos: Dict[str, int] = {name: 0 for name in function.blocks}
    iloop_header: Dict[str, Optional[str]] = {name: None for name in function.blocks}
    is_loop_header: Set[str] = set()
    irreducible_headers: Set[str] = set()
    reentry_edges: Set[Tuple[str, str]] = set()
    loop_edges: Set[Tuple[str, str]] = set()
    postorder: List[str] = []

    def tag_lhead(b: str, h: Optional[str]) -> None:
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

    def trav_loops_dfs(b0: str, pos: int) -> Optional[str]:
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
    block_layout = rpo

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

    # Parent header inference: the smallest containing loop that is not itself
    def parent_header_of(h: str) -> Optional[str]:
        candidates = [
            cand
            for cand, blocks in loop_membership.items()
            if h in blocks and cand != h
        ]
        if not candidates:
            return None
        # Choose the smallest loop that still contains the header (approximates immediacy)
        return min(candidates, key=lambda c: len(loop_membership[c]))

    parent_header_map: Dict[str, Optional[str]] = {
        h: parent_header_of(h) for h in loop_membership
    }

    # Build loops in header order by RPO index
    headers_sorted = sorted(loop_membership.keys(), key=lambda h: name_to_rpo_idx[h])
    loop_index_by_header: Dict[str, int] = {}
    loops: List[LoopInfo] = []
    for h in headers_sorted:
        blocks_in_loop = loop_membership[h]
        begin_idx = min(name_to_rpo_idx[b] for b in blocks_in_loop)
        end_idx = max(name_to_rpo_idx[b] for b in blocks_in_loop) + 1
        loop_index_by_header[h] = len(loops)
        loops.append(
            LoopInfo(
                header=h,
                level=0,  # filled later
                parent=None,  # filled later
                begin=begin_idx,
                end=end_idx,
                num_blocks=len(blocks_in_loop),
                irreducible=h in irreducible_headers,
                reentries=[],
            )
        )

    # Fill parent indices and levels
    for idx, loop in enumerate(loops):
        parent_header = parent_header_map.get(loop.header)
        if parent_header is not None and parent_header in loop_index_by_header:
            loop.parent = loop_index_by_header[parent_header]
        else:
            loop.parent = None

    def compute_level(idx: int, memo: Dict[int, int]) -> int:
        if idx in memo:
            return memo[idx]
        parent = loops[idx].parent
        if parent is None:
            memo[idx] = 0
        else:
            memo[idx] = compute_level(parent, memo) + 1
        return memo[idx]

    level_memo: Dict[int, int] = {}
    for i in range(len(loops)):
        loops[i].level = compute_level(i, level_memo)

    # Attach re-entry edges to owning loop (by header match)
    for src, dst in reentry_edges:
        header = iloop_header.get(dst)
        if header is None and dst in loop_membership:
            header = dst
        if header is not None and header in loop_index_by_header:
            loops[loop_index_by_header[header]].reentries.append((src, dst))

    # Build block_loop_map using innermost header
    block_loop_map: List[Optional[int]] = [None for _ in block_layout]
    for block_name in function.blocks:
        header = iloop_header.get(block_name)
        if header is None and block_name in loop_membership:
            header = block_name
        if header is None:
            continue
        loop_idx = loop_index_by_header.get(header)
        if loop_idx is None:
            continue
        block_idx = name_to_rpo_idx[block_name]
        block_loop_map[block_idx] = loop_idx

    # Reverse mapping: block -> loops containing it (innermost up the parent chain)
    block_to_loops: Dict[str, Set[int]] = {}
    for idx, block_name in enumerate(block_layout):
        loop_idx = block_loop_map[idx]
        chain: Set[int] = set()
        while loop_idx is not None:
            chain.add(loop_idx)
            loop_idx = loops[loop_idx].parent
        if chain:
            block_to_loops[block_name] = chain

    # Loop depth and exit edges
    exit_edges: Set[Tuple[str, str]] = set()
    for block_name, block in function.blocks.items():
        loop_set = block_to_loops.get(block_name, set())
        block.loop_depth = len(loop_set)

        for succ in block.successors:
            src_loops = loop_set
            dst_loops = block_to_loops.get(succ, set())
            if src_loops and not src_loops.intersection(dst_loops):
                exit_edges.add((block_name, succ))
    return (
        block_layout,
        block_loop_map,
        loops,
        loop_edges,
        exit_edges,
        postorder,
        rpo,
        name_to_rpo_idx,
    )


def compute_loop_membership_from_map(
    block_layout: List[str], block_loop_map: List[Optional[int]], loops: List[LoopInfo]
) -> Dict[int, Set[str]]:
    """Build loop membership keyed by loop index, including ancestor loops."""
    membership: Dict[int, Set[str]] = {i: set() for i in range(len(loops))}
    for idx, block_name in enumerate(block_layout):
        loop_idx = block_loop_map[idx]
        visited: Set[int] = set()
        while loop_idx is not None and loop_idx not in visited:
            membership.setdefault(loop_idx, set()).add(block_name)
            visited.add(loop_idx)
            loop_idx = loops[loop_idx].parent
    return membership


def compute_loop_children(loops: List[LoopInfo]) -> Dict[int, List[int]]:
    children: Dict[int, List[int]] = {i: [] for i in range(len(loops))}
    for idx, loop in enumerate(loops):
        if loop.parent is not None:
            children.setdefault(loop.parent, []).append(idx)
    return children


def compute_initial_liveness(
    function: Function,
    exit_edges: Set[Tuple[str, str]],
    postorder: List[str],
) -> None:
    """
    Phase 1: Compute initial liveness sets using postorder traversal of FL(G).

    Args:
        function: The Function object
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
            adjusted_val = val
            if (block_name, succ) in exit_edges:
                adjusted_val += 10**9
            # Exclude phi destinations from successor's live_in
            if val_idx in succ_block.phi_defs:
                # For phi destinations, find the incoming value from this block
                phi_instr = val_as_phi(function, val_idx)
                if phi_instr:
                    incoming_val = phi_instr.incoming_val_for_block(block_name)
                    val_idx = function.value_indices[incoming_val]
            if val_idx not in live_out:
                live_out[val_idx] = adjusted_val
            else:
                live_out[val_idx] = min(live_out[val_idx], adjusted_val)

    block.live_out = live_out

    # Accumulate live_in during reverse traversal; defer set filtering to the end
    live_in_work: Dict[int, float] = dict(block.live_out)

    # Initialize per-value use position collection
    block.next_use_distances_by_val = {}

    # Process instructions in reverse to compute actual use distances
    block_len = len(block.instructions)

    # Initialize liveness tracking for pressure calculation
    live_set = set(block.live_out.keys())
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
                    # Update live_in distance for this use
                    live_in_work[use_idx] = min(
                        live_in_work.get(use_idx, float("inf")), i
                    )
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
                    # Update live_in distance for this use
                    live_in_work[use_idx] = 0
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

    # LiveIn(B) = (LiveOut(B) - DEF(B)) ∪ USE(B), apply set ops once at the end
    block.live_in = {
        val_idx: val
        for val_idx, val in live_in_work.items()
        if val_idx not in block.def_set and val_idx not in block.phi_defs
    }
    for val_idx in block.use_set:
        block.live_in[val_idx] = min(
            block.live_in.get(val_idx, float("inf")),
            live_in_work.get(val_idx, float("inf")),
        )

    # Adjust live_in distances for pass-through variables
    for val_idx, dist in block.live_in.items():
        if dist >= block_len and val_idx in block.live_out:
            block.live_in[val_idx] = block.live_out[val_idx] + block_len

    # Finalize next_use_distances_by_val: sort and append live_out distances
    for val_idx, use_positions in block.next_use_distances_by_val.items():
        # Sort to get chronological order (since we collected in reverse)
        use_positions.sort()
        # Add liveout distance as the last entry (using final live_out values after propagation)
        if val_idx in block.live_out:
            # live_out[val_idx] is distance from block exit, so add block_len to get distance from start
            use_positions.append(block_len + block.live_out[val_idx])
        else:
            # Not live out, append infinity as the last entry
            use_positions.append(math.inf)

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
    block_layout: List[str],
    block_loop_map: List[Optional[int]],
    loops: List[LoopInfo],
    loop_edges: Set[Tuple[str, str]],
    exit_edges: Set[Tuple[str, str]],
    postorder: List[str],
    rpo: List[str],
) -> None:
    """
    Combined phase: Propagate liveness within loop bodies and compute next-use distances.

    Args:
        function: The Function object
        block_layout: List of blocks in RPO
        block_loop_map: Loop index per block (innermost)
        loops: Loop descriptors
        loop_edges: Set of loop edges
        exit_edges: Set of loop exit edges for distance penalties
        postorder: Postorder traversal of the reduced CFG FL(G)
        rpo: List of block names in reverse postorder (for index-to-name lookup)
    """

    # Track which blocks need distance recomputation
    affected_blocks: Set[str] = set()
    loop_membership = compute_loop_membership_from_map(
        block_layout, block_loop_map, loops
    )
    children = compute_loop_children(loops)

    def gather_blocks(loop_idx: int, acc: Set[str]) -> None:
        """Collect blocks for a loop and its nested children."""
        acc.update(loop_membership.get(loop_idx, set()))
        for child in children.get(loop_idx, []):
            gather_blocks(child, acc)

    def loop_tree_dfs(loop_idx: int) -> None:
        """Recursive traversal mirroring Algorithm 3."""
        loop_blocks_all: Set[str] = set()
        gather_blocks(loop_idx, loop_blocks_all)
        if not loop_blocks_all:
            return

        loop_live_vars: Set[int] = set()
        for block_name in loop_blocks_all:
            block = function.blocks[block_name]
            loop_live_vars.update(block.live_in.keys())
            loop_live_vars.update(block.live_out.keys())

        # Propagate placeholders to all blocks in this loop and nested loops
        for block_name in loop_blocks_all:
            block = function.blocks[block_name]
            for val_idx in loop_live_vars:
                if val_idx not in block.def_set and val_idx not in block.phi_defs:
                    block.live_in[val_idx] = block.live_in.get(val_idx, float("inf"))
                block.live_out[val_idx] = block.live_out.get(val_idx, float("inf"))
            affected_blocks.add(block_name)

        # Recurse into child loops
        for child in children.get(loop_idx, []):
            loop_tree_dfs(child)

    # Start from root loops (parent is None)
    root_indices = [idx for idx, loop in enumerate(loops) if loop.parent is None]
    for root in root_indices:
        loop_tree_dfs(root)

    # Compute loop-aware next-use distances using BFS (inner loops first)
    if loops:
        loop_order = sorted(
            range(len(loops)), key=lambda i: loops[i].level, reverse=True
        )

        for loop_idx in loop_order:
            blocks = loop_membership.get(loop_idx, set())
            header = loops[loop_idx].header
            if not blocks or header not in function.blocks:
                continue

            header_block = function.blocks[header]
            loop_live_vars = set(header_block.live_in.keys())
            loop_live_vars.update(header_block.live_out.keys())

            # Also collect live vars from nested loops/blocks
            nested_blocks: Set[str] = set()
            gather_blocks(loop_idx, nested_blocks)
            for blk_name in nested_blocks:
                blk = function.blocks[blk_name]
                loop_live_vars.update(blk.live_in.keys())
                loop_live_vars.update(blk.live_out.keys())

            min_dist: Dict[int, float] = {}
            visited: Set[str] = set()
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
            for val_idx, d in min_dist.items():
                if (
                    val_idx not in header_block.def_set
                    and val_idx not in header_block.phi_defs
                ):
                    if d < header_block.live_in.get(val_idx, math.inf):
                        header_block.live_in[val_idx] = d

    for block_name in affected_blocks:
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


def compute_liveness(
    function: Function,
) -> Tuple[
    List[str],
    List[Optional[int]],
    List[LoopInfo],
    Set[Tuple[str, str]],
    Set[Tuple[str, str]],
]:
    """
    Main function that orchestrates the two-phase liveness analysis.

    Args:
        function: The Function object to analyze

    Returns:
        (block_layout, block_loop_map, loops, loop_edges, exit_edges)
    """
    # Phase 0: Setup
    compute_predecessors_and_use_def_sets(function)

    # Build loop forest and identify loop edges
    (
        block_layout,
        block_loop_map,
        loops,
        loop_edges,
        exit_edges,
        postorder,
        rpo,
        name_to_rpo_idx,
    ) = build_loop_forest(function)

    # Phase 1: Initial liveness computation
    compute_initial_liveness(function, exit_edges, postorder)

    # Phase 2: Loop propagation and distance computation
    propagate_loop_liveness_and_distances(
        function,
        block_layout,
        block_loop_map,
        loops,
        loop_edges,
        exit_edges,
        postorder,
        rpo,
    )

    return block_layout, block_loop_map, loops, loop_edges, exit_edges
