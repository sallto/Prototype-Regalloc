"""
IR Parser for Simple SSA-like Intermediate Representation

Grammar:
- Function header: "function <name>" (no colon)
- Blank lines allowed between sections
- Block header: "block <name>:" (with colon)
- Instructions are indented by exactly two spaces
- op instructions: "op" followed by optional "uses=%v0,%v1" and/or "defs=%v2,%v3" clauses
  - Multiple uses/defs can appear on the same line, separated by spaces
  - Values are comma-separated with no spaces around commas
- jmp instructions: "jmp <block>[,<block>...]" (comma-separated targets, no spaces)
- phi instructions: "phi <dest> [<block>, <value>, <block>, <value> ...]"
  - Bracket contents are comma+space-separated pairs
- Identifiers: % followed by letters, numbers, underscores (e.g., %v0, %result_1)
"""

from ir import *
from typing import Optional
import liveness
import min_algorithm
import dominators
from liveness import get_next_use_distance
from min_algorithm import SpillReload
from coloring import color_program
import argparse
import sys



def parse_function(text: str) -> Function:
    """Parse IR text into a Function object."""
    lines = text.splitlines()
    line_no = 0

    # Skip empty lines at the beginning
    while line_no < len(lines) and lines[line_no].strip() == "":
        line_no += 1

    if line_no >= len(lines):
        raise ParseError("Empty input", line_no + 1)

    # Parse function header
    func_line = lines[line_no].strip()
    if not func_line.startswith("function "):
        raise ParseError(f"Expected 'function <name>', got '{func_line}'", line_no + 1)
    func_name = func_line[len("function "):].strip()
    if not func_name:
        raise ParseError("Function name cannot be empty", line_no + 1)

    function = Function(func_name)
    current_block = None
    value_idx = 0  # Counter for assigning unique indices to values
    line_no += 1

    while line_no < len(lines):
        line = lines[line_no]
        line_no += 1

        # Skip empty lines and comment lines
        if line.strip() == "":
            continue
        if line.strip().startswith(";"):
            continue

        # Check for block header
        if line.startswith("block "):
            if not line.endswith(":"):
                raise ParseError(f"Block header must end with ':', got '{line}'", line_no)
            block_name = line[len("block "):-1].strip()
            if not block_name:
                raise ParseError("Block name cannot be empty", line_no)
            if block_name in function.blocks:
                raise ParseError(f"Duplicate block name '{block_name}'", line_no)

            current_block = Block(block_name)
            function.add_block(current_block)
            continue

        # Check for instruction (must be indented by exactly two spaces)
        if line.startswith("  ") and not line.startswith("   "):
            if current_block is None:
                raise ParseError("Instruction found before any block", line_no)

            instr_line = line[2:]  # Remove the two spaces
            if instr_line.startswith("op "):
                op = parse_op_line(instr_line, line_no)
                current_block.instructions.append(op)
                # Assign indices to newly defined values
                for def_val in op.defs:
                    if def_val not in function.value_indices:
                        function.value_indices[def_val] = value_idx
                        value_idx += 1
            elif instr_line.startswith("jmp "):
                jmp = parse_jmp_line(instr_line, line_no)
                current_block.instructions.append(jmp)
                current_block.successors = jmp.targets  # Set block successors
            elif instr_line.startswith("phi "):
                phi = parse_phi_line(instr_line, line_no)
                current_block.instructions.append(phi)
                # Assign index to phi destination value
                if phi.dest not in function.value_indices:
                    function.value_indices[phi.dest] = value_idx
                    value_idx += 1
            else:
                raise ParseError(f"Unknown instruction type: '{instr_line}'", line_no)
            continue

        # If we get here, the line is malformed
        raise ParseError(f"Unexpected line format: '{line}'", line_no)

    if not function.blocks:
        raise ParseError("Function must have at least one block", len(lines))

    return function


def parse_op_line(line: str, line_no: int) -> Op:
    """Parse an op instruction line like 'op uses=%v0 defs=%v1'."""
    parts = line.split()
    if len(parts) < 1 or parts[0] != "op":
        raise ParseError(f"Expected 'op', got '{line}'", line_no)

    uses = []
    defs = []

    for part in parts[1:]:
        if part.startswith("uses="):
            uses_str = part[len("uses="):]
            if not uses_str:
                raise ParseError("Empty uses list", line_no)
            uses = uses_str.split(",")
        elif part.startswith("defs="):
            defs_str = part[len("defs="):]
            if not defs_str:
                raise ParseError("Empty defs list", line_no)
            defs = defs_str.split(",")
        else:
            raise ParseError(f"Unexpected op parameter: '{part}'", line_no)

    return Op(uses=uses, defs=defs)


def parse_jmp_line(line: str, line_no: int) -> Jump:
    """Parse a jmp instruction line like 'jmp b1,b2'."""
    parts = line.split()
    if len(parts) != 2 or parts[0] != "jmp":
        raise ParseError(f"Expected 'jmp <targets>', got '{line}'", line_no)

    targets_str = parts[1]
    if not targets_str:
        raise ParseError("Jump targets cannot be empty", line_no)

    targets = targets_str.split(",")
    if not all(targets):
        raise ParseError("Jump targets cannot be empty", line_no)

    return Jump(targets=targets)


def parse_phi_line(line: str, line_no: int) -> Phi:
    """Parse a phi instruction line like 'phi %v6 [b0, %v1, b1, %v4]'."""
    parts = line.split()
    if len(parts) < 3 or parts[0] != "phi":
        raise ParseError(f"Expected 'phi <dest> [<incomings>]', got '{line}'", line_no)

    dest = parts[1]
    if not dest.startswith("%"):
        raise ParseError(f"Phi destination must start with '%', got '{dest}'", line_no)

    # Find the bracket content
    bracket_start = line.find("[")
    bracket_end = line.find("]")
    if bracket_start == -1 or bracket_end == -1 or bracket_end < bracket_start:
        raise ParseError("Phi instruction missing brackets", line_no)

    bracket_content = line[bracket_start + 1:bracket_end].strip()
    if not bracket_content:
        raise ParseError("Phi incoming list cannot be empty", line_no)

    # Parse comma+space separated pairs
    items = [item.strip() for item in bracket_content.split(", ")]
    if len(items) % 2 != 0:
        raise ParseError("Phi incoming list must have even number of items (block,value pairs)", line_no)

    incomings = []
    for i in range(0, len(items), 2):
        block = items[i]
        value = items[i + 1]
        if not value.startswith("%"):
            raise ParseError(f"Phi value must start with '%', got '{value}'", line_no)
        incomings.append(PhiIncoming(block=block, value=value))

    return Phi(dest=dest, incomings=incomings)




def get_var_color(function: Function, val_idx: int) -> Optional[int]:
    """
    Get the color assigned to a variable by finding its defining instruction.
    
    Args:
        function: The Function object
        val_idx: Value index of the variable
        
    Returns:
        The color assigned to the variable, or None if not found
    """
    from ir import get_val_name
    var_name = get_val_name(function, val_idx)
    
    # Search through all blocks to find where this variable is defined
    for block in function.blocks.values():
        # Check phi instructions first
        for phi in block.phis():
            if phi.dest == var_name:
                return phi.dest_color
        
        # Check regular instructions
        for instr in block.instructions:
            if isinstance(instr, Op):
                if var_name in instr.defs:
                    return instr.def_colors.get(var_name)
    
    return None


def print_function(function: Function, idom: dict = None, dom_tree: dict = None) -> None:
    """Pretty-print a parsed Function."""
    print(f"Function: {function.name}")
    print("Blocks:")
    for block_name, block in function.blocks.items():
        print(f"  {block_name}:")
        print(f"    Predecessors: {block.predecessors}")
        print(f"    Successors: {block.successors}")
        if idom is not None:
            immediate_dom = idom.get(block_name)
            if immediate_dom is not None:
                print(f"    Immediate Dominator: {immediate_dom}")
            else:
                print(f"    Immediate Dominator: (entry block)")
        if dom_tree is not None:
            children = dom_tree.get(block_name, [])
            if children:
                print(f"    Dominator Tree Children: {sorted(children)}")
        # Convert val_idx to names for display
        from ir import get_val_name
        use_set_names = sorted([get_val_name(function, idx) for idx in block.use_set]) if block.use_set else []
        def_set_names = sorted([get_val_name(function, idx) for idx in block.def_set]) if block.def_set else []
        phi_uses_names = sorted([get_val_name(function, idx) for idx in block.phi_uses]) if block.phi_uses else []
        phi_defs_names = sorted([get_val_name(function, idx) for idx in block.phi_defs]) if block.phi_defs else []
        print(f"    USE set: {use_set_names if use_set_names else 'empty'}")
        print(f"    DEF set: {def_set_names if def_set_names else 'empty'}")
        print(f"    PhiUses: {phi_uses_names if phi_uses_names else 'empty'}")
        print(f"    PhiDefs: {phi_defs_names if phi_defs_names else 'empty'}")
        print(f"    Max Register Pressure: {block.max_register_pressure}")
        live_in_str = ', '.join(
            f"{get_val_name(function, val_idx)}:{dist}"
            if dist != U32_MAX
            else f"{get_val_name(function, val_idx)}:inf"
            for val_idx, dist in sorted(block.live_in.items())
        ) if isinstance(block.live_in, dict) and block.live_in else 'empty'
        live_out_str = ', '.join(
            f"{get_val_name(function, val_idx)}:{dist}"
            if dist != U32_MAX
            else f"{get_val_name(function, val_idx)}:inf"
            for val_idx, dist in sorted(block.live_out.items())
        ) if isinstance(block.live_out, dict) and block.live_out else 'empty'
        print(f"    LiveIn: {{{live_in_str}}}")
        print(f"    LiveOut: {{{live_out_str}}}")
        print("    Instructions:")
        # Process phi instructions first (they're always at the beginning)
        phi_count = 0
        for i, phi in enumerate(block.phis()):
            incomings_str = ", ".join(f"{inc.block}={inc.value}" for inc in phi.incomings)
            print(f"      {i}: phi {phi.dest} [{incomings_str}]")
            phi_count += 1
        
        # Process remaining instructions
        for i, instr in enumerate(block.instructions[phi_count:], start=phi_count):
            if isinstance(instr, Op):
                uses_str = ", ".join(instr.uses) if instr.uses else "none"
                defs_str = ", ".join(instr.defs) if instr.defs else "none"

                # Show next-use distances for values in this instruction
                next_use_str = ""
                if hasattr(block, 'next_use_distances_by_val') and block.next_use_distances_by_val:
                    next_uses = []
                    for var in sorted(set(instr.uses + instr.defs)):
                        if var in function.value_indices:
                            val_idx = function.value_indices[var]
                            dist = get_next_use_distance(block, val_idx, i, function)
                        else:
                            dist = U32_MAX
                        dist_str = "inf" if dist == U32_MAX else str(dist)
                        next_uses.append(f"{var}:{dist_str}")
                    if next_uses:
                        next_use_str = f" next_use={{{', '.join(next_uses)}}}"

                print(f"      {i}: op uses=[{uses_str}] defs=[{defs_str}]{next_use_str}")
            elif isinstance(instr, Jump):
                targets_str = ", ".join(instr.targets)
                print(f"      {i}: jmp {targets_str}")
        print()


def print_dominator_tree(function: Function, idom: dict, dom_tree: dict = None) -> None:
    """
    Print the dominator tree in a tree-like format.
    
    Args:
        function: The Function object
        idom: Dictionary mapping blocks to their immediate dominators
        dom_tree: Optional dictionary mapping blocks to their children in the dominator tree.
                 If None, will be computed from idom.
    """
    if dom_tree is None:
        dom_tree = dominators.build_dominator_tree(idom)
    
    # Find the root (entry block - has itself as idom or None)
    root = None
    for block_name, dom in idom.items():
        if dom is None or dom == block_name:
            root = block_name
            break
    
    if root is None:
        print("Dominator Tree: (no entry block found)")
        return
    
    print("Dominator Tree:")
    print("=" * 50)
    
    def print_tree_node(block_name: str, prefix: str = "", is_last: bool = True):
        """Recursively print the dominator tree."""
        # Print current node
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{block_name}")
        
        # Update prefix for children
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        # Get children and sort for consistent output
        children = sorted(dom_tree.get(block_name, []))
        
        # Print children
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            print_tree_node(child, child_prefix, is_last_child)
    
    print_tree_node(root)
    print()
    
    # Also print immediate dominators in a table format
    print("Immediate Dominators:")
    print("-" * 50)
    for block_name in sorted(function.blocks.keys()):
        dom = idom.get(block_name)
        if dom is None or dom == block_name:
            print(f"  {block_name:10s} -> (entry block)")
        else:
            print(f"  {block_name:10s} -> {dom}")
    print()


def print_function_with_spills(function: Function, spills_reloads: Dict[str, List[SpillReload]]) -> None:
    """
    Print the function IR with spill and reload instructions inserted at correct positions.

    Args:
        function: The Function object to print
        spills_reloads: Dictionary mapping block names to lists of SpillReload operations
    """
    print(f"function {function.name}")
    print()

    # Process blocks in the order they appear in function.blocks
    for block_name, block in function.blocks.items():
        print(f"block {block_name}:")

        # Get all operations for this block (already sorted by position)
        operations = spills_reloads.get(block_name, [])

        # Use an iterator to process operations in order
        op_iter = iter(operations)
        next_op = None
        try:
            next_op = next(op_iter)
        except StopIteration:
            next_op = None

        # Process phi instructions first (they're always at the beginning)
        phi_count = 0
        for instr_idx, phi in enumerate(block.phis()):
            # Print all operations that should appear before this instruction (position == instr_idx)
            while next_op is not None and next_op.position == instr_idx:
                var_name = get_val_name(function, next_op.val_idx)
                print(f"  {next_op.type} {var_name}")
                try:
                    next_op = next(op_iter)
                except StopIteration:
                    next_op = None
            
            incomings_str = ", ".join(f"{inc.block}, {inc.value}" for inc in phi.incomings)
            print(f"  phi {phi.dest} [{incomings_str}]")
            phi_count += 1
        
        # Process remaining instructions
        for instr_idx, instr in enumerate(block.instructions[phi_count:], start=phi_count):
            # Print all operations that should appear before this instruction (position == instr_idx)
            while next_op is not None and next_op.position == instr_idx:
                var_name = get_val_name(function, next_op.val_idx)
                print(f"  {next_op.type} {var_name}")
                try:
                    next_op = next(op_iter)
                except StopIteration:
                    next_op = None

            # Print the original instruction
            if isinstance(instr, Op):
                uses_str = ",".join(instr.uses) if instr.uses else ""
                defs_str = ",".join(instr.defs) if instr.defs else ""
                parts = []
                if uses_str:
                    parts.append(f"uses={uses_str}")
                if defs_str:
                    parts.append(f"defs={defs_str}")
                op_line = "op"
                if parts:
                    op_line += " " + " ".join(parts)
                print(f"  {op_line}")
            elif isinstance(instr, Jump):
                targets_str = ",".join(instr.targets)
                print(f"  jmp {targets_str}")

        # Handle operations after the last instruction (position >= len(block.instructions))
        while next_op is not None:
            var_name = get_val_name(function, next_op.val_idx)
            print(f"  {next_op.type} {var_name}")
            try:
                next_op = next(op_iter)
            except StopIteration:
                next_op = None

        print()


def print_function_with_colors(function: Function, spills_reloads: Dict[str, List[SpillReload]]) -> None:
    """
    Print the function IR with register colors shown at definition points and reloads.
    
    Args:
        function: The Function object to print
        spills_reloads: Dictionary mapping block names to lists of SpillReload operations
    """
    print(f"function {function.name}")
    print()

    # Process blocks in the order they appear in function.blocks
    for block_name, block in function.blocks.items():
        print(f"block {block_name}:")

        # Get all operations for this block (already sorted by position)
        operations = spills_reloads.get(block_name, [])

        # Use an iterator to process operations in order
        op_iter = iter(operations)
        next_op = None
        try:
            next_op = next(op_iter)
        except StopIteration:
            next_op = None

        # Process phi instructions first (they're always at the beginning)
        phi_count = 0
        for instr_idx, phi in enumerate(block.phis()):
            # Print all operations that should appear before this instruction (position == instr_idx)
            while next_op is not None and next_op.position == instr_idx:
                var_name = get_val_name(function, next_op.val_idx)
                if next_op.type == "reload":
                    # Show color for reload operations - use color from spill_reload
                    color = next_op.color
                    if color is not None:
                        print(f"  {next_op.type} {var_name} -> r{color}")
                    else:
                        print(f"  {next_op.type} {var_name}")
                else:
                    # Spill operations - show color being spilled from
                    color = next_op.color
                    if color is not None:
                        print(f"  {next_op.type} {var_name} <- r{color}")
                    else:
                        print(f"  {next_op.type} {var_name}")
                try:
                    next_op = next(op_iter)
                except StopIteration:
                    next_op = None
            
            # Show color for phi destination
            dest_color = phi.dest_color
            dest_str = phi.dest
            if dest_color is not None:
                dest_str = f"{phi.dest}->r{dest_color}"
            
            incomings_str = ", ".join(f"{inc.block}, {inc.value}" for inc in phi.incomings)
            print(f"  phi {dest_str} [{incomings_str}]")
            phi_count += 1
        
        # Process remaining instructions
        for instr_idx, instr in enumerate(block.instructions[phi_count:], start=phi_count):
            # Print all operations that should appear before this instruction (position == instr_idx)
            while next_op is not None and next_op.position == instr_idx:
                var_name = get_val_name(function, next_op.val_idx)
                if next_op.type == "reload":
                    # Show color for reload operations - use color from spill_reload
                    color = next_op.color
                    if color is not None:
                        print(f"  {next_op.type} {var_name} -> r{color}")
                    else:
                        print(f"  {next_op.type} {var_name}")
                else:
                    # Spill operations - show color being spilled from
                    color = next_op.color
                    if color is not None:
                        print(f"  {next_op.type} {var_name} <- r{color}")
                    else:
                        print(f"  {next_op.type} {var_name}")
                try:
                    next_op = next(op_iter)
                except StopIteration:
                    next_op = None

            # Print the original instruction with colors at def points
            if isinstance(instr, Op):
                uses_str = ",".join(instr.uses) if instr.uses else ""
                defs_str = ",".join(instr.defs) if instr.defs else ""
                
                # Add colors to defs
                if defs_str:
                    defs_with_colors = []
                    for def_var in instr.defs:
                        color = instr.def_colors.get(def_var)
                        if color is not None:
                            defs_with_colors.append(f"{def_var}->r{color}")
                        else:
                            defs_with_colors.append(def_var)
                    defs_str = ",".join(defs_with_colors)
                # INSERT_YOUR_CODE
                # Add colors to uses as well (similarly to defs)
                if uses_str:
                    uses_with_colors = []
                    for use_var in instr.uses:
                        color = instr.use_colors.get(use_var) if hasattr(instr, "use_colors") else None
                        if color is not None:
                            uses_with_colors.append(f"{use_var}<-r{color}")
                        else:
                            uses_with_colors.append(use_var)
                    uses_str = ",".join(uses_with_colors)
                parts = []
                if uses_str:
                    parts.append(f"uses={uses_str}")
                if defs_str:
                    parts.append(f"defs={defs_str}")
                op_line = "op"
                if parts:
                    op_line += " " + " ".join(parts)
                print(f"  {op_line}")
            elif isinstance(instr, Jump):
                targets_str = ",".join(instr.targets)
                print(f"  jmp {targets_str}")

        # Handle operations after the last instruction (position >= len(block.instructions))
        while next_op is not None:
            var_name = get_val_name(function, next_op.val_idx)
            if next_op.type == "reload":
                # Show color for reload operations - use color from spill_reload
                color = next_op.color
                if color is not None:
                    print(f"  {next_op.type} {var_name} -> r{color}")
                else:
                    print(f"  {next_op.type} {var_name}")
            else:
                # Spill operations - show color being spilled from
                color = next_op.color
                if color is not None:
                    print(f"  {next_op.type} {var_name} <- r{color}")
                else:
                    print(f"  {next_op.type} {var_name}")
            try:
                next_op = next(op_iter)
            except StopIteration:
                next_op = None

        print()


def print_function_with_register_state(function: Function, spills_reloads: Dict[str, List[SpillReload]], 
                                       k: int) -> None:
    """
    Print the function IR with register state columns on the left and colored IR on the right.
    
    Args:
        function: The Function object to print
        spills_reloads: Dictionary mapping block names to lists of SpillReload operations
        k: Number of available registers
    """
    from min_algorithm import is_last_use, topological_order
    
    # Determine column width for register columns
    # Each register column should be wide enough for variable names (e.g., "%v0")
    # Use a fixed width of 8 characters per register column
    reg_col_width = 8
    
    def format_register_state(reg_state: List[str]) -> str:
        """Format register state as a string with fixed-width columns."""
        reg_state_parts = []
        for i in range(k):
            var = reg_state[i] if i < len(reg_state) else None
            reg_state_parts.append((var if var else "").ljust(reg_col_width))
        return "".join(reg_state_parts) + " |"
    
    # Print header row
    header_parts = []
    for i in range(k):
        header_parts.append(f"R{i}".ljust(reg_col_width))
    header = "".join(header_parts) + " |"
    print(header)
    
    # Track final register state for each block
    block_final_states = {}
    
    # Process blocks in topological order so predecessors are processed before successors
    block_order = topological_order(function)
    
    # Process blocks in topological order
    for block_name in block_order:
        block = function.blocks[block_name]
        
        # Initialize register state for this block
        register_state = [None] * k
        
        # Initialize from predecessor states if this is not an entry block
        if block.predecessors:
            # If single predecessor, inherit its final state
            if len(block.predecessors) == 1:
                pred_name = block.predecessors[0]
                if pred_name in block_final_states:
                    register_state = block_final_states[pred_name].copy()
            else:
                # Multiple predecessors: compute intersection
                # A variable is in a register only if it's in the same register in ALL predecessors
                pred_states = []
                for pred_name in block.predecessors:
                    if pred_name in block_final_states:
                        pred_states.append(block_final_states[pred_name])
                
                if pred_states:
                    # Start with first predecessor's state
                    register_state = pred_states[0].copy()
                    # Intersect with other predecessors
                    for pred_state in pred_states[1:]:
                        for reg_idx in range(k):
                            if register_state[reg_idx] != pred_state[reg_idx]:
                                register_state[reg_idx] = None
        
        # Apply spills and reloads that happen at position 0 (before first instruction)
        from ir import get_val_name
        operations = spills_reloads.get(block_name, [])
        for op in operations:
            if op.position == 0:
                val_idx = op.val_idx
                var_name = get_val_name(function, val_idx)
                color = op.color  # Use color from spill_reload
                if color is not None:
                    reg_idx = color
                    if op.type == "reload":
                        register_state[reg_idx] = var_name
                    elif op.type == "spill":
                        if register_state[reg_idx] == var_name:
                            register_state[reg_idx] = None
        
        # Print block header
        reg_state_str = format_register_state(register_state)
        print(f"{reg_state_str}  block {block_name}:")
        
        # Get all operations for this block (already sorted by position)
        operations = spills_reloads.get(block_name, [])
        
        # Use an iterator to process operations in order
        op_iter = iter(operations)
        next_op = None
        try:
            next_op = next(op_iter)
        except StopIteration:
            next_op = None
        
        # Process phi instructions first (they're always at the beginning)
        phi_count = 0
        for instr_idx, phi in enumerate(block.phis()):
            # Print all operations that should appear before this instruction (position == instr_idx)
            while next_op is not None and next_op.position == instr_idx:
                # Update register state based on operation
                val_idx = next_op.val_idx
                var_name = get_val_name(function, val_idx)
                color = next_op.color  # Use color from spill_reload
                if color is not None:
                    reg_idx = color
                    if next_op.type == "spill":
                        if register_state[reg_idx] == var_name:
                            register_state[reg_idx] = None
                    elif next_op.type == "reload":
                        register_state[reg_idx] = var_name
                
                # Format and print
                reg_state_str = format_register_state(register_state)
                
                if next_op.type == "reload":
                    # Show color for reload operations
                    if color is not None:
                        print(f"{reg_state_str}  {next_op.type} {var_name} -> r{color}")
                    else:
                        print(f"{reg_state_str}  {next_op.type} {var_name}")
                else:
                    # Spill operations - show color being spilled from
                    if color is not None:
                        print(f"{reg_state_str}  {next_op.type} {var_name} <- r{color}")
                    else:
                        print(f"{reg_state_str}  {next_op.type} {var_name}")
                
                try:
                    next_op = next(op_iter)
                except StopIteration:
                    next_op = None
            
            # Process phi instruction
            instr_uses = []
            instr_defs = [phi.dest]
            
            # Process uses: if last use, free the register
            for use_var in instr_uses:
                if use_var in function.value_indices:
                    use_val_idx = function.value_indices[use_var]
                    if is_last_use(use_val_idx, block, instr_idx, function):
                        color = get_var_color(function, use_val_idx)
                        if color is not None:
                            reg_idx = color
                            if register_state[reg_idx] == use_var:
                                register_state[reg_idx] = None

            # Process defs: variable enters its assigned register
            for def_var in instr_defs:
                if def_var in function.value_indices:
                    def_val_idx = function.value_indices[def_var]
                    color = get_var_color(function, def_val_idx)
                    if color is not None:
                        reg_idx = color
                        register_state[reg_idx] = def_var
            
            # Format register state columns
            reg_state_str = format_register_state(register_state)
            
            # Show color for phi destination
            dest_color = phi.dest_color
            dest_str = phi.dest
            if dest_color is not None:
                dest_str = f"{phi.dest}->r{dest_color}"
            
            incomings_str = ", ".join(f"{inc.block}, {inc.value}" for inc in phi.incomings)
            print(f"{reg_state_str}  phi {dest_str} [{incomings_str}]")
            phi_count += 1
        
        # Process remaining instructions
        for instr_idx, instr in enumerate(block.instructions[phi_count:], start=phi_count):
            # Print all operations that should appear before this instruction (position == instr_idx)
            while next_op is not None and next_op.position == instr_idx:
                # Update register state based on operation
                val_idx = next_op.val_idx
                var_name = get_val_name(function, val_idx)
                color = next_op.color  # Use color from spill_reload
                if color is not None:
                    reg_idx = color
                    if next_op.type == "spill":
                        if register_state[reg_idx] == var_name:
                            register_state[reg_idx] = None
                    elif next_op.type == "reload":
                        register_state[reg_idx] = var_name
                
                # Format and print
                reg_state_str = format_register_state(register_state)
                
                if next_op.type == "reload":
                    # Show color for reload operations
                    if color is not None:
                        print(f"{reg_state_str}  {next_op.type} {var_name} -> r{color}")
                    else:
                        print(f"{reg_state_str}  {next_op.type} {var_name}")
                else:
                    # Spill operations - show color being spilled from
                    if color is not None:
                        print(f"{reg_state_str}  {next_op.type} {var_name} <- r{color}")
                    else:
                        print(f"{reg_state_str}  {next_op.type} {var_name}")
                
                try:
                    next_op = next(op_iter)
                except StopIteration:
                    next_op = None
            
            # Get uses and defs for this instruction
            if isinstance(instr, Op):
                instr_uses = instr.uses
                instr_defs = instr.defs
            else:
                instr_uses = []
                instr_defs = []
            
            # Process uses: if last use, free the register
            for use_var in instr_uses:
                if use_var in function.value_indices:
                    use_val_idx = function.value_indices[use_var]
                    if is_last_use(use_val_idx, block, instr_idx, function):
                        color = get_var_color(function, use_val_idx)
                        if color is not None:
                            reg_idx = color
                            if register_state[reg_idx] == use_var:
                                register_state[reg_idx] = None

            # Process defs: variable enters its assigned register
            for def_var in instr_defs:
                if def_var in function.value_indices:
                    def_val_idx = function.value_indices[def_var]
                    color = get_var_color(function, def_val_idx)
                    if color is not None:
                        reg_idx = color
                        register_state[reg_idx] = def_var
            
            # Format register state columns
            reg_state_str = format_register_state(register_state)
            
            # Print the original instruction with colors at def points
            if isinstance(instr, Op):
                uses_str = ",".join(instr.uses) if instr.uses else ""
                defs_str = ",".join(instr.defs) if instr.defs else ""
                
                # Add colors to defs
                if defs_str:
                    defs_with_colors = []
                    for def_var in instr.defs:
                        color = instr.def_colors.get(def_var)
                        if color is not None:
                            defs_with_colors.append(f"{def_var}->r{color}")
                        else:
                            defs_with_colors.append(def_var)
                    defs_str = ",".join(defs_with_colors)
                
                parts = []
                if uses_str:
                    parts.append(f"uses={uses_str}")
                if defs_str:
                    parts.append(f"defs={defs_str}")
                op_line = "op"
                if parts:
                    op_line += " " + " ".join(parts)
                print(f"{reg_state_str}  {op_line}")
            elif isinstance(instr, Jump):
                targets_str = ",".join(instr.targets)
                print(f"{reg_state_str}  jmp {targets_str}")
        
        # Handle operations after the last instruction (position >= len(block.instructions))
        while next_op is not None:
            # Update register state based on operation
            val_idx = next_op.val_idx
            var_name = get_val_name(function, val_idx)
            color = next_op.color  # Use color from spill_reload
            if color is not None:
                reg_idx = color
                if next_op.type == "spill":
                    if register_state[reg_idx] == var_name:
                        register_state[reg_idx] = None
                elif next_op.type == "reload":
                    register_state[reg_idx] = var_name
            
            # Format and print
            reg_state_str = format_register_state(register_state)
            
            if next_op.type == "reload":
                # Show color for reload operations
                if color is not None:
                    print(f"{reg_state_str}  {next_op.type} {var_name} -> r{color}")
                else:
                    print(f"{reg_state_str}  {next_op.type} {var_name}")
            else:
                # Spill operations - show color being spilled from
                if color is not None:
                    print(f"{reg_state_str}  {next_op.type} {var_name} <- r{color}")
                else:
                    print(f"{reg_state_str}  {next_op.type} {var_name}")
            try:
                next_op = next(op_iter)
            except StopIteration:
                next_op = None
        
        # Store final register state for this block
        block_final_states[block_name] = register_state.copy()
        
        print()


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

        for val_idx in block.live_out:
            found_in_successor = False
            successor_live_ins = []
            var_name = get_val_name(function, val_idx)

            for successor in block.successors:
                if successor in function.blocks:
                    succ_block = function.blocks[successor]
                    # Convert val_idx to names for error message
                    succ_live_in_names = sorted([get_val_name(function, idx) for idx in succ_block.live_in])
                    successor_live_ins.append(
                        f"{successor}: {succ_live_in_names}"
                    )
                    if val_idx in succ_block.live_in:
                        found_in_successor = True
                        break
                    # Also check if var flows into a phi in this successor from this block
                    for phi in succ_block.phis():
                        incoming_val = phi.incoming_val_for_block(block_name)
                        if incoming_val == var_name:
                            found_in_successor = True
                            break
                    if found_in_successor:
                        break

            if not found_in_successor:
                # Convert val_idx to names for error message
                live_out_names = sorted([get_val_name(function, idx) for idx in block.live_out])
                error_msg = (
                    f"Variable '{var_name}' in LiveOut of block '{block_name}' "
                    f"not found in LiveIn of any successor.\n"
                    f"  LiveOut({block_name}): {live_out_names}\n"
                    f"  Successors LiveIn: {successor_live_ins}"
                )
                errors.append(error_msg)

    if errors:
        error_details = "\n\n".join(errors)
        raise AssertionError(f"Liveness correctness check failed:\n\n{error_details}")

    return True


def test_parser(ir_file: str, k: int = 3) -> None:
    """Test the parser with IR from the specified file."""
    try:
        with open(ir_file, "r") as f:
            content = f.read()

        # Skip the first line if it starts with a comment or if it's README.md
        lines = content.splitlines()
        if lines and (lines[0].startswith("#") or ir_file.endswith("README.md")):
            ir_text = "\n".join(lines[1:])  # Skip the header line
        else:
            ir_text = content

        print(f"Parsing {ir_file} IR...")
        function = parse_function(ir_text)
        print("Parse successful!")
        print()

        # Compute liveness
        print("Computing liveness analysis...")
        block_layout, block_loop_map, loops, loop_edges, exit_edges = liveness.compute_liveness(function)
        print("Liveness analysis completed!")


        # Check liveness correctness
        print("Checking liveness correctness...")
        try:
            check_liveness_correctness(function)
            print("Liveness correctness check passed!")
        except AssertionError as e:
            print(f"Liveness correctness check failed: {e}")
            return
        print()

        # Compute dominators
        print("Computing dominator tree...")
        idom = dominators.compute_dominators(function)
        dom_tree = dominators.build_dominator_tree(idom)
        print("Dominator tree computation completed!")
        print()
        
        # Print dominator tree
        print_dominator_tree(function, idom, dom_tree)

        # Run the Min algorithm for register allocation
        print(f"Running Min algorithm with k={k}...")
        spills_reloads = min_algorithm.min_algorithm(
            function, block_layout, block_loop_map, loops, k=k
        )
        print("Min algorithm completed!")
        print()

                # Print the IR with spills and reloads
        print(f"IR with Spills and Reloads (k={k}):")
        print("=" * 50)
        print_function_with_spills(function, spills_reloads)
        print()

        # Run SSA-based coloring
        print(f"Running SSA-based coloring with k={k}...")
        color_program(function, k=k, spills_reloads=spills_reloads)
        print("Coloring completed!")
        print()


        # Print the IR with colors at def points and reloads
        print(f"IR with Register Colors (k={k}):")
        print("=" * 50)
        print_function_with_colors(function, spills_reloads)
        print()
        
        # Print the IR with register state visualization
        print(f"IR with Register State Visualization (k={k}):")
        print("=" * 50)
        print_function_with_register_state(function, spills_reloads, k)

    except ParseError as e:
        print(f"Parse error: {e}")
    except FileNotFoundError:
        print(f"{ir_file} not found")


def main():
    parser = argparse.ArgumentParser(description="IR Parser and Liveness Analysis")
    parser.add_argument("file", help="Path to the IR file to parse")
    parser.add_argument("-k", "--registers", type=int, default=3,
                        help="Number of available registers (default: 3)")
    args = parser.parse_args()

    test_parser(args.file, args.registers)


if __name__ == "__main__":
    main()
