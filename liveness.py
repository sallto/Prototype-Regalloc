"""
Liveness Analysis for Strict SSA IR

Implements the non-iterative liveness analysis algorithm from:
"Non-iterative Data-Flow Analysis for Computing Liveness Sets in Strict SSA"

The algorithm has two phases:
1. Postorder traversal of FL(G) (CFG without loop edges)
2. Loop-nesting forest traversal to propagate liveness within loops
"""

from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from ir import Function, Block, Op, Phi


@dataclass
class LoopNode:
    """Represents a node in the loop-nesting forest."""
    block_name: str  # Name of the block (or loop header)
    is_loop: bool    # True if this represents a loop header
    children: List['LoopNode']
    parent: 'LoopNode' = None


def compute_predecessors(function: Function) -> None:
    """
    Build predecessor lists for each block from successor information.

    Args:
        function: The Function object with blocks
    """
    # Clear existing predecessors
    for block in function.blocks.values():
        block.predecessors = []

    # Build predecessors from successors
    for block_name, block in function.blocks.items():
        for successor in block.successors:
            if successor in function.blocks:
                function.blocks[successor].predecessors.append(block_name)
            else:
                raise ValueError(f"Block '{block_name}' has successor '{successor}' that doesn't exist")


def compute_use_def_sets(function: Function) -> None:
    """
    Compute USE, DEF, PhiUses, and PhiDefs sets for each block.

    Args:
        function: The Function object with blocks
    """
    from main import Op, Phi

    for block in function.blocks.values():
        # Clear existing sets
        block.use_set = set()
        block.def_set = set()
        block.phi_uses = set()
        block.phi_defs = set()

        for instr in block.instructions:
            if isinstance(instr, Op):
                # Add uses and defs from Op instructions
                block.use_set.update(instr.uses)
                block.def_set.update(instr.defs)
            elif isinstance(instr, Phi):
                # Add phi destination to phi_defs
                block.phi_defs.add(instr.dest)
                # Add all incoming values to phi_uses
                for incoming in instr.incomings:
                    block.phi_uses.add(incoming.value)


def build_loop_forest(function: Function) -> Tuple[Dict[str, LoopNode], Set[Tuple[str, str]]]:
    """
    Build loop-nesting forest for reducible graphs and identify loop edges.

    Args:
        function: The Function object with blocks

    Returns:
        Tuple of (loop_forest_dict, loop_edges_set)
        - loop_forest_dict: Maps block names to their LoopNode in the forest
        - loop_edges_set: Set of (source, target) tuples that are loop edges
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

    def dfs(block_name: str, path: Set[str]):
        """DFS traversal to find loops and build forest structure."""
        if block_name in visiting:
            # Found a back edge - this indicates a loop
            loop_headers.add(block_name)
            # The back edge is from current path to block_name
            # Find the source of the back edge
            for path_block in path:
                if function.blocks[path_block].successors.__contains__(block_name):
                    back_edges.add((path_block, block_name))
                    break
            return

        if block_name in visited:
            return

        visiting.add(block_name)
        path.add(block_name)

        for successor in function.blocks[block_name].successors:
            dfs(successor, path)

        visiting.remove(block_name)
        path.remove(block_name)
        visited.add(block_name)

        # Create LoopNode for this block
        loop_forest[block_name] = LoopNode(
            block_name=block_name,
            is_loop=block_name in loop_headers,
            children=[]
        )

    # Find a root block (one with no predecessors)
    root_candidates = [name for name, block in function.blocks.items()
                      if not block.predecessors]

    if not root_candidates:
        # If no blocks have no predecessors, pick any block as root
        root_candidates = list(function.blocks.keys())[:1]

    # Start DFS from each root candidate
    for root in root_candidates:
        if root not in visited:
            dfs(root, set())

    # Build parent-child relationships in the forest
    # For this simplified version, we'll create a basic tree structure
    # A full implementation would need proper loop nesting analysis

    # For now, create a simple forest where loop headers are parents of their loop bodies
    processed = set()
    loop_edges = set()

    def build_forest_hierarchy():
        """Build the hierarchical structure of the loop forest."""
        # This is a simplified hierarchy - real implementation needs proper nesting
        for block_name, node in loop_forest.items():
            if node.is_loop:
                # Find blocks that are in this loop
                # Simplified: any block that has a path to this header via back edges
                loop_blocks = set()
                for src, tgt in back_edges:
                    if tgt == block_name:
                        # Find all blocks dominated by this loop header
                        # Simplified: just add the source block
                        loop_blocks.add(src)

                for loop_block in loop_blocks:
                    if loop_block in loop_forest and loop_block != block_name:
                        loop_forest[block_name].children.append(loop_forest[loop_block])
                        loop_forest[loop_block].parent = loop_forest[block_name]

        # Add regular blocks as children of non-loop blocks
        for block_name, node in loop_forest.items():
            if not node.is_loop and not node.parent:
                # Find a parent - simplified: attach to first predecessor that's a loop header
                for pred in function.blocks[block_name].predecessors:
                    if pred in loop_forest and loop_forest[pred].is_loop:
                        loop_forest[pred].children.append(node)
                        node.parent = loop_forest[pred]
                        break

    build_forest_hierarchy()

    return loop_forest, back_edges


def get_reduced_cfg_edges(function: Function, loop_edges: Set[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Get edges of the reduced CFG FL(G) by removing loop edges.

    Args:
        function: The Function object
        loop_edges: Set of (source, target) tuples that are loop edges

    Returns:
        List of (source, target) tuples representing edges in FL(G)
    """
    edges = []
    for block_name, block in function.blocks.items():
        for successor in block.successors:
            edge = (block_name, successor)
            if edge not in loop_edges:
                edges.append(edge)
    return edges


def postorder_traversal(edges: List[Tuple[str, str]], all_blocks: Set[str]) -> List[str]:
    """
    Perform postorder traversal of the graph defined by edges.

    Args:
        edges: List of (source, target) tuples
        all_blocks: Set of all block names

    Returns:
        List of block names in postorder
    """
    # Build adjacency list
    adj = {}
    for block in all_blocks:
        adj[block] = []
    for src, tgt in edges:
        adj[src].append(tgt)

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


def compute_initial_liveness(function: Function, loop_forest: Dict[str, LoopNode],
                           loop_edges: Set[Tuple[str, str]]) -> None:
    """
    Phase 1: Compute initial liveness sets using postorder traversal of FL(G).

    Args:
        function: The Function object
        loop_forest: Loop forest structure
        loop_edges: Set of loop edges to exclude
    """
    # Get reduced CFG edges (without loop edges)
    reduced_edges = get_reduced_cfg_edges(function, loop_edges)
    all_blocks = set(function.blocks.keys())

    # Perform postorder traversal
    postorder = postorder_traversal(reduced_edges, all_blocks)

    # Initialize live sets (already done in Block.__init__)

    # Process blocks in postorder
    for block_name in postorder:
        block = function.blocks[block_name]

        # LiveOut(B) = union of LiveIn(S) for all successors S
        block.live_out = set()
        for successor in block.successors:
            if successor in function.blocks:
                succ_block = function.blocks[successor]
                block.live_out.update(succ_block.live_in)

        # Add PhiUses(B) to LiveOut(B)
        block.live_out.update(block.phi_uses)

        # Remove PhiDefs(S) from LiveIn(S) for each successor S
        for successor in block.successors:
            if successor in function.blocks:
                succ_block = function.blocks[successor]
                succ_block.live_in -= succ_block.phi_defs

        # LiveIn(B) = (LiveOut(B) - DEF(B)) ∪ USE(B)
        block.live_in = (block.live_out - block.def_set) | block.use_set

        # Add PhiDefs(B) to LiveIn(B)
        block.live_in.update(block.phi_defs)


def propagate_loop_liveness(function: Function, loop_forest: Dict[str, LoopNode]) -> None:
    """
    Phase 2: Propagate liveness within loop bodies using Algorithm 3.

    Args:
        function: The Function object
        loop_forest: Loop forest structure
    """
    def loop_tree_dfs(node: LoopNode) -> None:
        """Recursive DFS traversal of the loop forest (Algorithm 3)."""
        if node.is_loop:
            block_n = function.blocks[node.block_name]
            # LiveLoop = LiveIn(BN) \ PhiDefs(BN)
            live_loop = block_n.live_in - block_n.phi_defs

            # Visit children
            for child in node.children:
                block_m = function.blocks[child.block_name]
                # LiveIn(BM) = LiveIn(BM) ∪ LiveLoop
                block_m.live_in.update(live_loop)
                # LiveOut(BM) = LiveOut(BM) ∪ LiveLoop
                block_m.live_out.update(live_loop)

                # Recursively process child
                loop_tree_dfs(child)

    # Start from root nodes (nodes with no parent)
    roots = [node for node in loop_forest.values() if node.parent is None]
    for root in roots:
        loop_tree_dfs(root)


def compute_liveness(function: Function) -> None:
    """
    Main function that orchestrates the two-phase liveness analysis.

    Args:
        function: The Function object to analyze
    """
    # Phase 0: Setup
    compute_predecessors(function)
    compute_use_def_sets(function)

    # Build loop forest and identify loop edges
    loop_forest, loop_edges = build_loop_forest(function)

    # Phase 1: Initial liveness computation
    compute_initial_liveness(function, loop_forest, loop_edges)

    # Phase 2: Loop propagation
    propagate_loop_liveness(function, loop_forest)
