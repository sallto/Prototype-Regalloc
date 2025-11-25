"""
Dominator Tree Computation

Implements dominator analysis using the Cooper-Harvey-Kennedy algorithm.
A node d dominates a node n if every path from the entry node to n must go through d.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional
from ir import Function, Block


@dataclass
class DominatorInfo:
    """Stores dominator information for a block."""
    immediate_dominator: Optional[str]  # The immediate dominator (idom)
    children: List[str]  # Blocks that this block immediately dominates
    dominance_frontier: Set[str]  # Dominance frontier (optional, computed separately)


def find_entry_block(function: Function) -> Optional[str]:
    """
    Find the entry block (block with no predecessors).
    
    Args:
        function: The Function object
        
    Returns:
        Name of the entry block, or None if not found
    """
    for block_name, block in function.blocks.items():
        if not block.predecessors:
            return block_name
    # If no block has no predecessors, return the first block
    if function.blocks:
        return list(function.blocks.keys())[0]
    return None


def compute_postorder(function: Function, entry: str) -> List[str]:
    """
    Compute postorder traversal of the CFG starting from entry.
    
    Args:
        function: The Function object
        entry: Entry block name
        
    Returns:
        List of block names in postorder
    """
    visited = set()
    postorder = []
    
    def dfs(block_name: str):
        if block_name in visited:
            return
        visited.add(block_name)
        
        block = function.blocks.get(block_name)
        if block:
            for successor in block.successors:
                if successor in function.blocks:
                    dfs(successor)
        
        postorder.append(block_name)
    
    dfs(entry)
    return postorder


def intersect(
    idom: Dict[str, Optional[str]], 
    reverse_postorder: List[str],
    b1: str, 
    b2: str
) -> Optional[str]:
    """
    Find the nearest common dominator of b1 and b2.
    
    This implements the "intersect" operation from the Cooper-Harvey-Kennedy algorithm.
    It finds the first common dominator by walking up both idom chains, using reverse
    postorder numbers to determine which finger to advance.
    
    Args:
        idom: Dictionary mapping blocks to their immediate dominators
        reverse_postorder: List of blocks in reverse postorder (for numbering)
        b1: First block name
        b2: Second block name
        
    Returns:
        Name of the nearest common dominator, or None if not found
    """
    # Create reverse postorder number mapping
    rpo_num = {block: i for i, block in enumerate(reverse_postorder)}
    
    finger1 = b1
    finger2 = b2
    
    # Walk up both chains until they meet
    while finger1 != finger2:
        # Get reverse postorder numbers (higher number = later in reverse postorder)
        num1 = rpo_num.get(finger1, -1)
        num2 = rpo_num.get(finger2, -1)
        
        # Advance the finger with the higher number (further from entry)
        if num1 > num2:
            if finger1 is not None and finger1 in idom:
                finger1 = idom[finger1]
            else:
                return None
        elif num2 > num1:
            if finger2 is not None and finger2 in idom:
                finger2 = idom[finger2]
            else:
                return None
        else:
            # Both are at same level, advance both
            if finger1 is not None and finger1 in idom:
                finger1 = idom[finger1]
            else:
                finger1 = None
            if finger2 is not None and finger2 in idom:
                finger2 = idom[finger2]
            else:
                finger2 = None
            
            if finger1 is None and finger2 is None:
                return None
    
    return finger1


def compute_dominators(function: Function) -> Dict[str, Optional[str]]:
    """
    Compute immediate dominators using the Cooper-Harvey-Kennedy algorithm.
    
    Returns a dictionary mapping each block name to its immediate dominator.
    The entry block has itself as its immediate dominator.
    
    Args:
        function: The Function object with blocks and predecessors computed
        
    Returns:
        Dictionary mapping block names to their immediate dominator (or None for entry)
    """
    entry = find_entry_block(function)
    if entry is None:
        return {}
    
    # Compute reverse postorder (postorder reversed)
    postorder = compute_postorder(function, entry)
    reverse_postorder = list(reversed(postorder))
    
    # Initialize idom: entry dominates itself, others are undefined
    idom: Dict[str, Optional[str]] = {}
    for block_name in function.blocks.keys():
        if block_name == entry:
            idom[block_name] = entry
        else:
            idom[block_name] = None
    
    # Iterate until fixed point
    changed = True
    while changed:
        changed = False
        
        # Process blocks in reverse postorder (excluding entry)
        for block_name in reverse_postorder:
            if block_name == entry:
                continue
                
            block = function.blocks.get(block_name)
            if not block or not block.predecessors:
                continue
            
            # Find first defined predecessor
            new_idom = None
            for pred in block.predecessors:
                if pred in idom and idom[pred] is not None:
                    new_idom = pred
                    break
            
            if new_idom is None:
                continue
            
            # Intersect with all other predecessors
            for pred in block.predecessors:
                if pred != new_idom and pred in idom and idom[pred] is not None:
                    # Find intersection of new_idom and pred
                    intersect_result = intersect(idom, reverse_postorder, new_idom, pred)
                    if intersect_result is not None:
                        new_idom = intersect_result
            
            # Update if changed
            if idom[block_name] != new_idom:
                idom[block_name] = new_idom
                changed = True
    
    return idom


def dominates(idom: Dict[str, Optional[str]], d: str, n: str) -> bool:
    """
    Check if block d dominates block n.
    
    A block d dominates n if every path from the entry to n goes through d.
    This is checked by walking up the idom chain from n to see if d is encountered.
    
    Args:
        idom: Dictionary mapping blocks to their immediate dominators
        d: Dominator block name
        n: Block name to check
        
    Returns:
        True if d dominates n, False otherwise
    """
    # Every block dominates itself
    if d == n:
        return True
    
    # Walk up the idom chain from n
    current = n
    while current is not None and current in idom:
        current = idom[current]
        if current == d:
            return True
    
    return False


def build_dominator_tree(idom: Dict[str, Optional[str]]) -> Dict[str, List[str]]:
    """
    Build the dominator tree structure.
    
    The dominator tree is a tree where each node's children are the blocks
    it immediately dominates.
    
    Args:
        idom: Dictionary mapping blocks to their immediate dominators
        
    Returns:
        Dictionary mapping each block to a list of blocks it immediately dominates
    """
    tree: Dict[str, List[str]] = {}
    
    # Initialize all blocks with empty children lists
    for block_name in idom.keys():
        tree[block_name] = []
    
    # Build parent-child relationships
    for block_name, parent in idom.items():
        if parent is not None and parent != block_name:
            tree[parent].append(block_name)
    
    return tree


def compute_dominance_frontiers(
    function: Function, idom: Dict[str, Optional[str]]
) -> Dict[str, Set[str]]:
    """
    Compute the dominance frontier for each block.
    
    The dominance frontier of a block b is the set of blocks where b's dominance
    ends. More formally, a block y is in the dominance frontier of x if:
    - x dominates a predecessor of y, but
    - x does not strictly dominate y
    
    This is useful for phi placement in SSA construction.
    
    Args:
        function: The Function object
        idom: Dictionary mapping blocks to their immediate dominators
        
    Returns:
        Dictionary mapping each block to its dominance frontier (set of block names)
    """
    df: Dict[str, Set[str]] = {}
    
    # Initialize dominance frontiers
    for block_name in function.blocks.keys():
        df[block_name] = set()
    
    # For each block
    for block_name in function.blocks.keys():
        block = function.blocks.get(block_name)
        if not block:
            continue
        
        # If block has more than one predecessor
        if len(block.predecessors) > 1:
            # For each predecessor p of block
            for pred in block.predecessors:
                runner = pred
                # Walk up the idom chain from p until we reach block's idom
                while runner is not None and runner != idom.get(block_name):
                    df[runner].add(block_name)
                    if runner in idom:
                        runner = idom[runner]
                    else:
                        break
    
    return df

