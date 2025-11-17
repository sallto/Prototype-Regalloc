"""
Liveness Analysis for Strict SSA IR

Implements the non-iterative liveness analysis algorithm from:
"Non-iterative Data-Flow Analysis for Computing Liveness Sets in Strict SSA"

The algorithm has two phases:
1. Postorder traversal of FL(G) (CFG without loop edges)
2. Loop-nesting forest traversal to propagate liveness within loops
"""

# todo: non reducible control flow
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from ir import Function, Block, Op, Phi


@dataclass
class LoopNode:
    """Represents a node in the loop-nesting forest."""

    block_name: str  # Name of the block (or loop header)
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
        # Compute USE/DEF sets (order-aware)
        block.use_set = set()
        block.def_set = set()
        block.phi_uses = set()
        block.phi_defs = set()

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
) -> None:
    """
    Phase 1: Compute initial liveness sets using postorder traversal of FL(G).

    Args:
        function: The Function object
        loop_forest: Loop forest structure
        loop_edges: Set of loop edges to exclude
    """
    # Perform postorder traversal of reduced CFG (FL(G))
    postorder = postorder_traversal_reduced_cfg(function, loop_edges)

    # Initialize live sets (already done in Block.__init__)

    # Process blocks in postorder
    for block_name in postorder:
        block = function.blocks[block_name]

        # Remove PhiDefs(S) from LiveIn(S) for each successor S before computing LiveOut
        successor_live_ins = {}
        for successor in block.successors:
            if successor in function.blocks:
                succ_block = function.blocks[successor]
                # Store original LiveIn for successors
                successor_live_ins[successor] = succ_block.live_in.copy()
                # Remove phi definitions from LiveIn for propagation
                succ_block.live_in -= succ_block.phi_defs

        # LiveOut(B) = union of LiveIn(S) for all successors S (with phi defs removed)
        block.live_out = set()
        for successor in block.successors:
            if successor in function.blocks:
                succ_block = function.blocks[successor]
                block.live_out.update(succ_block.live_in)

        # LiveIn(B) = (LiveOut(B) - DEF(B)) ∪ USE(B)
        block.live_in = (block.live_out - block.def_set) | block.use_set

        # Add PhiDefs(B) to LiveIn(B)
        block.live_in.update(block.phi_defs)

        # Restore original LiveIn for successors (add back phi defs)
        for successor in block.successors:
            if successor in function.blocks:
                succ_block = function.blocks[successor]
                succ_block.live_in = successor_live_ins[successor]


def propagate_loop_liveness(
    function: Function, loop_forest: Dict[str, LoopNode]
) -> None:
    """
    Phase 2: Propagate liveness within loop bodies using Algorithm 3.

    Args:
        function: The Function object
        loop_forest: Loop forest structure
    """

    def loop_tree_dfs(node: LoopNode) -> None:
        """Recursive DFS traversal of the loop forest (Algorithm 3)."""
        if node.is_loop:
            # Collect all live variables in this loop
            loop_live_vars = set()

            # Add live vars from the loop header
            block_n = function.blocks[node.block_name]
            loop_live_vars.update(block_n.live_in)
            loop_live_vars.update(block_n.live_out)

            # Add live vars from all children (recursive)
            def collect_live_vars(n: LoopNode) -> None:
                block = function.blocks[n.block_name]
                loop_live_vars.update(block.live_in)
                loop_live_vars.update(block.live_out)
                for child in n.children:
                    collect_live_vars(child)

            for child in node.children:
                collect_live_vars(child)

            # Propagate all loop live vars to the header and children
            block_n.live_in.update(loop_live_vars)
            block_n.live_out.update(loop_live_vars)

            for child in node.children:
                block_m = function.blocks[child.block_name]
                block_m.live_in.update(loop_live_vars)
                block_m.live_out.update(loop_live_vars)

                # Recursively process child
                loop_tree_dfs(child)

    # Start from root nodes (nodes with no parent)
    roots = [node for node in loop_forest.values() if node.parent is None]
    for root in roots:
        loop_tree_dfs(root)


def propagate_next_use_distances(
    function: Function, loop_forest: Dict[str, LoopNode]
) -> None:
    """
    Propagate next-use distances within loop bodies to handle loop back edges.

    Similar to propagate_loop_liveness but works with distance dictionaries,
    taking minimum distances when merging.

    Args:
        function: The Function object
        loop_forest: Loop forest structure
    """

    def loop_tree_dfs(node: LoopNode) -> None:
        """Recursive DFS traversal of the loop forest."""
        if node.is_loop:
            # Collect all next-use distances in this loop
            loop_distances = {}

            # Add distances from the loop header
            block_n = function.blocks[node.block_name]
            for var, dist in block_n.live_in.items():
                if var not in loop_distances:
                    loop_distances[var] = dist
                else:
                    loop_distances[var] = min(loop_distances[var], dist)

            for var, dist in block_n.live_out.items():
                if var not in loop_distances:
                    loop_distances[var] = dist
                else:
                    loop_distances[var] = min(loop_distances[var], dist)

            # Add distances from all children (recursive)
            def collect_distances(n: LoopNode) -> None:
                block = function.blocks[n.block_name]
                for var, dist in block.live_in.items():
                    if var not in loop_distances:
                        loop_distances[var] = dist
                    else:
                        loop_distances[var] = min(loop_distances[var], dist)

                for var, dist in block.live_out.items():
                    if var not in loop_distances:
                        loop_distances[var] = dist
                    else:
                        loop_distances[var] = min(loop_distances[var], dist)

                for child in n.children:
                    collect_distances(child)

            for child in node.children:
                collect_distances(child)

            # Propagate minimum distances to the header and children
            # Update header
            for var, min_dist in loop_distances.items():
                block_n.live_in[var] = min_dist
                block_n.live_out[var] = min_dist

            # Update children
            for child in node.children:
                block_m = function.blocks[child.block_name]
                for var, min_dist in loop_distances.items():
                    block_m.live_in[var] = min_dist
                    block_m.live_out[var] = min_dist

                # Recursively process child
                loop_tree_dfs(child)

    # Start from root nodes (nodes with no parent)
    roots = [node for node in loop_forest.values() if node.parent is None]
    for root in roots:
        loop_tree_dfs(root)


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


def compute_next_use_distances(
    function: Function,
) -> Dict[Tuple[str, int], List[float]]:
    """
    Compute next-use distances for all variable definitions in linear IR.

    For each variable definition, calculates the instruction distances to all subsequent uses.
    Distance is infinity if there are no further uses.

    Args:
        function: The Function object with instructions

    Returns:
        Dict mapping (variable_name, definition_instruction_index) -> list of distances
    """
    distances = {}

    # For linear IR without control flow, process blocks in order
    # Assume blocks are processed in the order they appear (b0, b1, etc.)
    block_names = sorted(function.blocks.keys())

    # Collect all instructions in global order
    all_instructions = []
    for block_name in block_names:
        block = function.blocks[block_name]
        for instr in block.instructions:
            all_instructions.append((instr, block_name))

    # For each instruction that defines variables
    for def_idx, (instr, def_block) in enumerate(all_instructions):
        if isinstance(instr, Op) and instr.defs:
            for var in instr.defs:
                key = (var, instr.val_local_idx)
                distances[key] = []

                # Scan forward for uses of this variable
                for use_idx in range(def_idx + 1, len(all_instructions)):
                    use_instr, use_block = all_instructions[use_idx]

                    if isinstance(use_instr, Op) and var in use_instr.uses:
                        # Found a use - calculate distance
                        distance = use_instr.val_local_idx - instr.val_local_idx
                        distances[key].append(distance)

                # If no uses found, add infinity
                if not distances[key]:
                    distances[key].append(float("inf"))

    return distances


def compute_block_next_use_distances(function: Function) -> None:
    """
    Compute next-use distances for live_in and live_out sets by processing
    blocks in post-order and instructions in reverse within each block.

    Args:
        function: The Function object to analyze
    """
    # Build loop forest and identify loop edges (needed for postorder traversal)
    loop_forest, loop_edges, loop_membership = build_loop_forest(function)
    exit_edges = compute_loop_exit_edges(loop_membership, function)

    # Get postorder traversal of blocks
    postorder = postorder_traversal_reduced_cfg(function, loop_edges)

    print("exit_edges", exit_edges)
    print("postorder", postorder)
    for block_name in postorder:
        # Compute live_out as the merged live_in from all successors, taking minimums
        live_out = {}
        for succ in function.blocks[block_name].successors:
            if succ in function.blocks:
                succ_live_in = function.blocks[succ].live_in
                for var, val in succ_live_in.items():
                    if var not in live_out:
                        live_out[var] = val
                    else:
                        live_out[var] = min(live_out[var], val)
        function.blocks[block_name].live_out = live_out

        i = len(function.blocks[block_name].instructions)
        for instr in reversed(function.blocks[block_name].instructions):
            if isinstance(instr, Op):
                for use in instr.uses:
                    if use in function.blocks[block_name].live_in:
                        function.blocks[block_name].live_in[use] = min(
                            function.blocks[block_name].live_in[use], i
                        )
            i -= 1
        for var, dist in function.blocks[block_name].live_in.items():
            if dist >= len(function.blocks[block_name].instructions) and var in function.blocks[block_name].live_out:
                function.blocks[block_name].live_in[var] = function.blocks[
                    block_name
                ].live_out[var] + len(function.blocks[block_name].instructions)


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
    compute_initial_liveness(function, loop_forest, loop_edges)

    # Phase 2: Loop propagation
    propagate_loop_liveness(function, loop_forest)

    for block_name in function.blocks:
        function.blocks[block_name].live_in = {
            var: float("inf") for var in function.blocks[block_name].live_in
        }
        function.blocks[block_name].live_out = {
            var: float("inf") for var in function.blocks[block_name].live_out
        }

    # Phase 3: Compute next-use distances
    compute_block_next_use_distances(function)

    # Phase 4: Propagate next-use distances within loops
    propagate_next_use_distances(function, loop_forest)
