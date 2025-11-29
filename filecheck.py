#!/usr/bin/env python3
"""
FileCheck-like tool for IR register allocation output.

Supports verification and auto-generation of CHECK directives for:
- Spills and reloads (CHECK-SPILL)
- Register colors (CHECK-COLOR)
"""

import re
import subprocess
import sys
import argparse
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import liveness
from liveness import get_next_use_distance
from main import parse_function


class CheckType(Enum):
    """Type of CHECK directive."""
    SPILL = "SPILL"
    COLOR = "COLOR"


@dataclass
class CheckDirective:
    """Represents a CHECK directive."""
    check_type: CheckType
    is_next: bool  # True for CHECK-*-NEXT directives
    is_not: bool   # True for CHECK-*-NOT directives
    pattern: str   # The pattern to match
    k: Optional[int]  # Number of registers (None if not specified)
    line_number: int  # Line number in IR file where this check appears
    original_line: str  # Original line from IR file


@dataclass
class SpillIRInstruction:
    """Represents an instruction in the SPILL IR."""
    kind: str  # "op", "spill", "reload", "jmp", "phi"
    uses: List[str] = field(default_factory=list)
    defs: List[str] = field(default_factory=list)
    variable: Optional[str] = None  # For spill/reload
    targets: List[str] = field(default_factory=list)  # For jmp
    line_number: int = 0  # Original line number in output


@dataclass
class SpillIRBlock:
    """Represents a block in the SPILL IR."""
    name: str
    instructions: List[SpillIRInstruction] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)


@dataclass
class SpillIRFunction:
    """Represents a function parsed from SPILL section."""
    name: str
    blocks: Dict[str, SpillIRBlock] = field(default_factory=dict)


@dataclass
class RegisterState:
    """Represents register and spill state at a program point."""
    R: Set[str] = field(default_factory=set)  # Variables in registers
    S: Set[str] = field(default_factory=set)  # Variables in spill slots


@dataclass
class EvictionEvent:
    """Tracks a deferred eviction event where we're uncertain which variable was evicted."""
    candidates: Set[str]  # Variables that could have been evicted (R ∩ S)
    reload_var: str       # The variable being reloaded that triggered this
    line_number: int      # Line number for error reporting
    block_name: str       # Block name for error reporting


def parse_checks(ir_content: str) -> Dict[CheckType, List[CheckDirective]]:
    """
    Parse CHECK directives from IR file content.
    
    Returns a dictionary mapping CheckType to list of CheckDirective objects.
    """
    checks: Dict[CheckType, List[CheckDirective]] = {
        CheckType.SPILL: [],
        CheckType.COLOR: []
    }
    
    lines = ir_content.splitlines()
    for line_num, line in enumerate(lines, start=1):
        # Look for CHECK directives: ; CHECK-<TYPE>[-K<k>][-NEXT][-NOT]: pattern
        # Support both old format (no K) and new format (with K)
        match = re.match(r'\s*;\s*CHECK-(\w+)(?:-K(\d+))?(-NEXT)?(-NOT)?:\s*(.*)', line)
        if match:
            check_type_str, k_str, next_flag, not_flag, pattern = match.groups()
            
            # Map string to CheckType
            check_type_map = {
                "SPILL": CheckType.SPILL,
                "COLOR": CheckType.COLOR
            }
            
            check_type = check_type_map.get(check_type_str)
            if check_type:
                k_value = int(k_str) if k_str else None
                directive = CheckDirective(
                    check_type=check_type,
                    is_next=next_flag is not None,
                    is_not=not_flag is not None,
                    pattern=pattern.strip(),
                    k=k_value,
                    line_number=line_num,
                    original_line=line
                )
                checks[check_type].append(directive)
    
    return checks


def parse_spill_ir(spill_lines: List[str]) -> SpillIRFunction:
    """
    Parse SPILL section IR into structured data.
    
    Args:
        spill_lines: List of lines from the SPILL section
        
    Returns:
        SpillIRFunction object with parsed blocks and instructions
    """
    function = None
    current_block = None
    line_num = 0
    
    for line in spill_lines:
        line_num += 1
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
        
        # Parse function header
        if stripped.startswith("function "):
            func_name = stripped[len("function "):].strip()
            function = SpillIRFunction(name=func_name)
            continue
        
        # Parse block header
        if stripped.startswith("block ") and stripped.endswith(":"):
            block_name = stripped[len("block "):-1].strip()
            if function is None:
                raise ValueError(f"Block '{block_name}' found before function declaration")
            current_block = SpillIRBlock(name=block_name)
            function.blocks[block_name] = current_block
            continue
        
        # Parse instructions (must be indented with 2 spaces)
        if line.startswith("  ") and not line.startswith("   "):
            if current_block is None:
                continue  # Skip instructions outside blocks
            
            instr_line = line[2:].strip()  # Remove 2-space indent
            
            # Parse spill instruction
            if instr_line.startswith("spill "):
                var = instr_line[len("spill "):].strip()
                current_block.instructions.append(
                    SpillIRInstruction(kind="spill", variable=var, line_number=line_num)
                )
            # Parse reload instruction
            elif instr_line.startswith("reload "):
                var_part = instr_line[len("reload "):].strip()
                # Handle "reload %vX -> rN" format (ignore color info)
                var = var_part.split("->")[0].strip()
                current_block.instructions.append(
                    SpillIRInstruction(kind="reload", variable=var, line_number=line_num)
                )
            # Parse op instruction
            elif instr_line.startswith("op "):
                uses = []
                defs = []
                parts = instr_line[len("op "):].split()
                for part in parts:
                    if part.startswith("uses="):
                        uses_str = part[len("uses="):]
                        if uses_str:
                            uses = uses_str.split(",")
                    elif part.startswith("defs="):
                        defs_str = part[len("defs="):]
                        if defs_str:
                            defs = defs_str.split(",")
                current_block.instructions.append(
                    SpillIRInstruction(kind="op", uses=uses, defs=defs, line_number=line_num)
                )
            # Parse jmp instruction
            elif instr_line.startswith("jmp "):
                targets_str = instr_line[len("jmp "):].strip()
                targets = targets_str.split(",") if targets_str else []
                current_block.instructions.append(
                    SpillIRInstruction(kind="jmp", targets=targets, line_number=line_num)
                )
                # Update block successors
                current_block.successors = targets
            # Parse phi instruction (if present)
            elif instr_line.startswith("phi "):
                # For now, just record it as a phi - we may need to parse incomings later
                parts = instr_line.split()
                if len(parts) >= 2:
                    dest = parts[1]
                    current_block.instructions.append(
                        SpillIRInstruction(kind="phi", defs=[dest], line_number=line_num)
                    )
    
    # Build predecessor lists
    if function:
        for block_name, block in function.blocks.items():
            for succ_name in block.successors:
                if succ_name in function.blocks:
                    if block_name not in function.blocks[succ_name].predecessors:
                        function.blocks[succ_name].predecessors.append(block_name)
    
    return function if function else SpillIRFunction(name="")


def parse_output_sections(output: str) -> Dict[CheckType, List[str]]:
    """
    Parse main.py output into sections for each CHECK type.
    
    Returns a dictionary mapping CheckType to list of output lines.
    """
    sections: Dict[CheckType, List[str]] = {
        CheckType.SPILL: [],
        CheckType.COLOR: []
    }
    
    lines = output.splitlines()
    current_section: Optional[CheckType] = None
    collecting = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Detect section headers
        if "IR with Spills and Reloads" in line:
            current_section = CheckType.SPILL
            collecting = False
            # Skip until we see the actual content (after separator line)
            i += 1
            # Look for separator line (=====)
            while i < len(lines) and "=" not in lines[i]:
                i += 1
            i += 1  # Skip separator
            collecting = True
            continue
        elif "IR with Register Colors" in line:
            current_section = CheckType.COLOR
            collecting = False
            i += 1
            while i < len(lines) and "=" not in lines[i]:
                i += 1
            i += 1
            collecting = True
            continue
        
        # Stop collecting if we hit another major section
        if collecting and current_section:
            # Check if this is start of another section we care about
            if ("IR with Spills and Reloads" in line or 
                "IR with Register Colors" in line):
                # Will be handled in next iteration
                collecting = False
                current_section = None
                i += 1
                continue
            
            # Stop COLOR section when we hit Register State Visualization
            if current_section == CheckType.COLOR and "IR with Register State Visualization" in line:
                collecting = False
                current_section = None
                i += 1
                continue
            
            # Stop if we hit a blank line followed by non-IR content
            # or another major header
            if (line.strip() == "" and i + 1 < len(lines) and 
                ("Function:" in lines[i+1] or "Register Color" in lines[i+1] or
                 "Register State" in lines[i+1] or
                 "Dominator Tree" in lines[i+1] or "Parsing" in lines[i+1] or
                 "Computing" in lines[i+1] or "Running" in lines[i+1])):
                collecting = False
                current_section = None
                i += 1
                continue
            
            # Collect the line
            sections[current_section].append(line)
        
        i += 1
    
    return sections


def match_pattern(pattern: str, line: str) -> bool:
    """
    Match a CHECK pattern against a line.
    
    Patterns support regex-like matching similar to FileCheck.
    Special handling for common patterns and whitespace flexibility.
    """
    # Normalize whitespace: collapse multiple spaces/tabs to single space
    pattern_stripped = pattern.strip()
    line_stripped = line.strip()
    
    # Normalize whitespace more aggressively
    pattern_normalized = re.sub(r'\s+', ' ', pattern_stripped)
    line_normalized = re.sub(r'\s+', ' ', line_stripped)
    
    # Try literal substring match (allows partial matches)
    if pattern_normalized in line_normalized:
        return True
    
    # Try exact match after normalization
    if pattern_normalized == line_normalized:
        return True
    
    # Try regex match if pattern contains regex-like constructs
    # Replace common FileCheck patterns:
    # {{.*}} -> .* (general matching)
    regex_pattern = pattern_normalized
    regex_pattern = regex_pattern.replace('{{.*}}', '.*')
    regex_pattern = regex_pattern.replace('{{.*?}}', '.*?')
    
    try:
        if re.search(regex_pattern, line_normalized, re.IGNORECASE):
            return True
    except re.error:
        pass
    
    # Try as regex directly
    try:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    except re.error:
        pass
    
    return False


def verify_checks(checks: Dict[CheckType, List[CheckDirective]], 
                 sections: Dict[CheckType, List[str]], 
                 k: int,
                 file_name: str) -> Tuple[bool, List[str], List[str]]:
    """
    Verify that all CHECK directives match the output sections.
    Also detect missing checks (patterns in output without CHECK directives).
    Only checks directives that match the specified k value (or have no k specified).
    
    Args:
        checks: Dictionary of check types to directives
        sections: Dictionary of check types to output lines
        k: Number of registers to filter checks by
        file_name: Name of the IR file being checked
    
    Returns (success, list of error messages, list of warnings about missing checks).
    """
    errors = []
    warnings = []
    
    for check_type, directives in checks.items():
        # Filter directives to only those matching k (or with no k specified for backward compatibility)
        filtered_directives = [d for d in directives if d.k is None or d.k == k]
        
        if not filtered_directives:
            continue
        section_lines = sections.get(check_type, [])
        
        if not section_lines and directives:
            errors.append(f"{file_name}: No output section found for {check_type.value} checks")
            continue
        
        current_pos = 0  # Position in section_lines
        matched_lines = set()  # Track which output lines were matched
        
        for directive in filtered_directives:
            if directive.is_not:
                # Check that pattern does NOT appear after current position
                found = False
                for i in range(current_pos, len(section_lines)):
                    if match_pattern(directive.pattern, section_lines[i]):
                        found = True
                        break
                if found:
                    errors.append(
                        f"{file_name}:{directive.line_number}: CHECK-{check_type.value}-NOT pattern "
                        f"'{directive.pattern}' found in output"
                    )
            elif directive.is_next:
                # Pattern must match the very next line
                if current_pos >= len(section_lines):
                    errors.append(
                        f"{file_name}:{directive.line_number}: CHECK-{check_type.value}-NEXT pattern "
                        f"'{directive.pattern}' expected but reached end of output"
                    )
                else:
                    if match_pattern(directive.pattern, section_lines[current_pos]):
                        matched_lines.add(current_pos)
                        current_pos += 1
                    else:
                        errors.append(
                            f"{file_name}:{directive.line_number}: CHECK-{check_type.value}-NEXT pattern "
                            f"'{directive.pattern}' does not match line:\n"
                            f"  {section_lines[current_pos] if current_pos < len(section_lines) else '(end of output)'}"
                        )
                        # Don't advance current_pos on failure - this is a strict error
            else:
                # Pattern must appear somewhere after current position
                found = False
                for i in range(current_pos, len(section_lines)):
                    if match_pattern(directive.pattern, section_lines[i]):
                        matched_lines.add(i)
                        current_pos = i + 1
                        found = True
                        break
                
                if not found:
                    errors.append(
                        f"{file_name}:{directive.line_number}: CHECK-{check_type.value} pattern "
                        f"'{directive.pattern}' not found in output"
                    )
        
        # Detect missing checks: find important patterns in output that weren't matched
        important_patterns = detect_important_patterns(section_lines, check_type)
        for line_idx, pattern in important_patterns.items():
            if line_idx not in matched_lines:
                # Check if this pattern is covered by any existing CHECK directive
                # by checking if any directive pattern matches this output line
                is_covered = False
                for directive in directives:
                    if not directive.is_not:
                        # Check if the directive pattern matches this output line/pattern
                        if match_pattern(directive.pattern, pattern):
                            is_covered = True
                            break
                        # Also check if the directive pattern matches the full line
                        if line_idx < len(section_lines):
                            full_line = section_lines[line_idx]
                            if match_pattern(directive.pattern, full_line):
                                is_covered = True
                                break
                
                if not is_covered:
                    warnings.append(
                        f"{file_name}: Missing CHECK-{check_type.value} directive for: {pattern.strip()}"
                    )
    
    return len(errors) == 0, errors, warnings


def detect_important_patterns(section_lines: List[str], check_type: CheckType) -> Dict[int, str]:
    """
    Detect important patterns in output that should have CHECK directives.
    
    Returns a dictionary mapping line index to the pattern string.
    """
    important = {}
    
    for idx, line in enumerate(section_lines):
        line_stripped = line.strip()
        
        # Skip empty lines, headers, and separators
        if not line_stripped or line_stripped.startswith("function") or \
           line_stripped.startswith("block") or line_stripped.startswith("=") or \
           (line_stripped.startswith("R") and "|" in line_stripped):
            continue
        
        # For SPILL section: detect spills and reloads (most critical)
        if check_type == CheckType.SPILL:
            if line_stripped.startswith("spill ") or line_stripped.startswith("reload "):
                important[idx] = line_stripped
        
        # For COLOR section: detect color assignments (->rN) and operations with colors
        elif check_type == CheckType.COLOR:
            if "->r" in line_stripped:
                important[idx] = line_stripped
            elif line_stripped.startswith("reload ") or line_stripped.startswith("spill "):
                important[idx] = line_stripped
            elif line_stripped.startswith("op ") and "->r" in line_stripped:
                important[idx] = line_stripped
    
    return important


def generate_checks(sections: Dict[CheckType, List[str]], 
                   ir_content: str, k: int) -> str:
    """
    Generate CHECK directives from output sections and insert them into IR content.
    
    Returns updated IR content with CHECK directives added.
    """
    lines = ir_content.splitlines()
    result_lines = []
    
    # Remove existing CHECK directives from input
    filtered_lines = []
    for line in lines:
        if not re.match(r'\s*;\s*CHECK-', line):
            filtered_lines.append(line)
    
    result_lines.extend(filtered_lines)
    result_lines.append("")
    result_lines.append(f"; CHECK directives (auto-generated for k={k})")
    
    # Generate SPILL checks
    if sections.get(CheckType.SPILL):
        result_lines.append("")
        result_lines.append(f"; CHECK-SPILL-K{k} directives:")
        for line in sections[CheckType.SPILL]:
            line = line.strip()
            # Skip empty lines, headers, and separators
            if (line and 
                not line.startswith("function") and 
                not line.startswith("block") and
                not line.startswith("=") and
                line != ""):
                # Extract meaningful content (skip leading whitespace from output)
                pattern = line.lstrip()
                result_lines.append(f"; CHECK-SPILL-K{k}: {pattern}")
    
    # Generate COLOR checks
    if sections.get(CheckType.COLOR):
        result_lines.append("")
        result_lines.append(f"; CHECK-COLOR-K{k} directives:")
        for line in sections[CheckType.COLOR]:
            line = line.strip()
            # Skip empty lines, headers, separators, and non-IR content
            if (line and 
                not line.startswith("function") and 
                not line.startswith("block") and
                not line.startswith("=") and
                "IR with" not in line and
                not line.startswith("R") and
                "|" not in line and
                line != ""):
                pattern = line.lstrip()
                result_lines.append(f"; CHECK-COLOR-K{k}: {pattern}")
    
    return "\n".join(result_lines)


def topological_order_spill_ir(function: SpillIRFunction) -> List[str]:
    """
    Return blocks in topological order (predecessors before successors).
    
    Args:
        function: The SpillIRFunction object
        
    Returns:
        List of block names in topological order
    """
    # Find entry blocks (blocks with no predecessors)
    entry_blocks = [name for name, block in function.blocks.items() if not block.predecessors]
    
    # If no blocks have predecessors, pick the first block as entry
    if not entry_blocks:
        entry_blocks = [list(function.blocks.keys())[0]] if function.blocks else []
    
    visited = set()
    order = []
    
    def dfs(block_name: str):
        if block_name in visited:
            return
        visited.add(block_name)
        
        # Visit all successors
        block = function.blocks.get(block_name)
        if block:
            for successor in block.successors:
                if successor in function.blocks:
                    dfs(successor)
        
        order.append(block_name)
    
    # Start DFS from each entry block
    for entry in entry_blocks:
        dfs(entry)
    
    # Reverse to get topological order (predecessors first)
    return order[::-1]


@dataclass
class RegPressureError:
    """Represents a register pressure error."""
    error_type: str
    message: str
    block_name: str
    line_number: int
    variable: Optional[str] = None


def is_variable_live(block_name: str, var: str, instr_idx: int, function, spill_function: SpillIRFunction) -> bool:
    """
    Check if a variable is live at a specific instruction point in a block.
    
    Args:
        block_name: Name of the block in the spill IR
        var: Variable name to check
        instr_idx: Index of the instruction in the spill IR (0-based within the block)
        function: The Function object with liveness information (from original IR)
        spill_function: The SpillIRFunction object (to map block names)
        
    Returns:
        True if variable is live at this point, False otherwise
    """
    # Find the corresponding block in the original IR
    # Block names should match between spill IR and original IR
    if block_name not in function.blocks:
        # If block not found, assume variable is live (conservative)
        return True
    
    block = function.blocks[block_name]
    
    # Map instruction index from spill IR to original IR
    # Count only non-spill/reload instructions up to instr_idx
    if block_name not in spill_function.blocks:
        # If spill block not found, assume variable is live (conservative)
        return True
    
    spill_block = spill_function.blocks[block_name]
    original_instr_idx = 0
    for i in range(min(instr_idx, len(spill_block.instructions))):
        spill_instr = spill_block.instructions[i]
        # Only count op, jmp, and phi instructions (skip spill/reload)
        if spill_instr.kind in ("op", "jmp", "phi"):
            original_instr_idx += 1
    
    # Check next-use distance first - if finite, variable is live at this point
    # Use the mapped instruction index from original IR
    next_use_dist = get_next_use_distance(block, var, original_instr_idx, function)
    if next_use_dist != float('inf'):
        return True
    
    # Check if variable is live-out (needed for successor blocks)
    # This is important even if next-use distance is infinite (variable might be needed later)
    if isinstance(block.live_out, dict):
        if var in block.live_out:
            return True
    elif isinstance(block.live_out, set):
        if var in block.live_out:
            return True
    
    # Variable has no finite next use and is not live-out - it's dead
    return False


def remove_dead_variables_from_r(state: RegisterState, block_name: str, instr_idx: int, 
                                 function, spill_function: SpillIRFunction) -> None:
    """
    Remove dead variables from R set based on liveness analysis.
    
    Args:
        state: The RegisterState to update
        block_name: Name of the current block
        instr_idx: Index of the current instruction (0 = block entry)
        function: The Function object with liveness information (None if not available)
        spill_function: The SpillIRFunction object
    """
    if function is None:
        # No liveness information available, skip
        return
    
    if block_name not in function.blocks:
        # Block not found in original IR, skip
        return
    
    block = function.blocks[block_name]
    
    # Create a copy of R to iterate over (since we'll be modifying it)
    vars_to_check = list(state.R)
    for var in vars_to_check:
        # Check if variable is used anywhere in the current block
        # (not just in use_set, which only includes uses before defs)
        used_in_block = var in block.use_set or var in block.phi_uses
        if not used_in_block:
            # Check all instructions to see if variable is used anywhere
            for instr in block.instructions:
                if hasattr(instr, 'uses') and var in instr.uses:
                    used_in_block = True
                    break
                # Also check phi incomings from this block
                if hasattr(instr, 'incomings'):
                    for incoming in instr.incomings:
                        if incoming.block == block_name and incoming.value == var:
                            used_in_block = True
                            break
                    if used_in_block:
                        break
        
        # Also check if variable is defined in this block - if so, it can't be evicted
        # if it's used after definition (even if not in use_set)
        defined_in_block = var in block.def_set or var in block.phi_defs
        if defined_in_block:
            # Variable is defined in this block - check if it's used after definition
            found_def = False
            for instr in block.instructions:
                # Check for def
                if hasattr(instr, 'defs') and var in instr.defs:
                    found_def = True
                elif hasattr(instr, 'dest') and instr.dest == var:  # Phi
                    found_def = True
                # Check for use after def
                if found_def:
                    if hasattr(instr, 'uses') and var in instr.uses:
                        used_in_block = True
                        break
            # If variable is defined in this block, don't evict it even if not used
            # (it might be used later in the same block, and we're checking at an early point)
            if not used_in_block:
                # Still mark as used to prevent eviction - variables defined in block
                # should stay in R until we're sure they're not needed
                used_in_block = True
        
        # Check if variable is used as a phi input in any successor block
        # If so, it can't be evicted because phis need their input values
        used_as_phi_input = False
        for succ_name in block.successors:
            if succ_name not in function.blocks:
                continue
            succ_block = function.blocks[succ_name]
            # Check if variable is used as a phi input from this block
            for instr in succ_block.instructions:
                if hasattr(instr, 'incomings'):
                    for incoming in instr.incomings:
                        if incoming.block == block_name and incoming.value == var:
                            used_as_phi_input = True
                            break
                    if used_as_phi_input:
                        break
            if used_as_phi_input:
                break
        
        # If variable is not used in current block and not used as phi input, check if it can be evicted
        # A variable can be evicted if all paths from this block redefine it before use
        if not used_in_block and not used_as_phi_input:
            # Check if variable is redefined in all successor blocks before any use
            can_evict = True
            for succ_name in block.successors:
                if succ_name not in function.blocks:
                    can_evict = False
                    break
                succ_block = function.blocks[succ_name]
                # Check if variable is redefined before use in successor
                # Scan instructions to see if def comes before use
                found_def = False
                found_use_before_def = False
                for instr in succ_block.instructions:
                    # Check for use (excluding phi inputs, which we already checked)
                    if hasattr(instr, 'uses') and var in instr.uses:
                        if not found_def:
                            found_use_before_def = True
                            break
                    # Check for def
                    if hasattr(instr, 'defs') and var in instr.defs:
                        found_def = True
                    elif hasattr(instr, 'dest') and instr.dest == var:  # Phi
                        found_def = True
                
                # If variable is used before being redefined, can't evict
                if found_use_before_def:
                    can_evict = False
                    break
                # If variable is not redefined in this successor, check if it flows to blocks that need it
                if not found_def:
                    # Variable flows through this successor - need to check if it's needed
                    # For now, be conservative: if not redefined, assume it's needed
                    can_evict = False
                    break
            
            if can_evict:
                # Variable can be evicted - remove it from R
                state.R.discard(var)
                continue
        
        # Variable is used in block or can't be evicted - check if it's live at this point
        is_live = False
        
        # If variable is defined in this block, check if it's used at or after the current instruction
        # Variables defined in the current block should stay in R if they're used later
        if defined_in_block:
            # Map instruction index to original IR
            spill_block = spill_function.blocks[block_name]
            original_instr_idx = 0
            for i in range(min(instr_idx, len(spill_block.instructions))):
                spill_instr = spill_block.instructions[i]
                if spill_instr.kind in ("op", "jmp", "phi"):
                    original_instr_idx += 1
            
            # Check if variable is used at or after this instruction in the original IR
            for i in range(original_instr_idx, len(block.instructions)):
                instr = block.instructions[i]
                if hasattr(instr, 'uses') and var in instr.uses:
                    is_live = True
                    break
            
            # Also check if variable is live-out
            if isinstance(block.live_out, dict):
                if var in block.live_out:
                    is_live = True
            elif isinstance(block.live_out, set):
                if var in block.live_out:
                    is_live = True
        else:
            # Variable not defined in this block - use normal liveness check
            # At block entry (instr_idx == 0), check live_in and live_out
            if instr_idx == 0:
                # Check live_in
                if isinstance(block.live_in, dict):
                    if var in block.live_in:
                        is_live = True
                elif isinstance(block.live_in, set):
                    if var in block.live_in:
                        is_live = True
                
                # Check live_out
                if isinstance(block.live_out, dict):
                    if var in block.live_out:
                        is_live = True
                elif isinstance(block.live_out, set):
                    if var in block.live_out:
                        is_live = True
            else:
                # For other instruction indices, use the full liveness check
                is_live = is_variable_live(block_name, var, instr_idx, function, spill_function)
        
        if not is_live:
            # Variable is dead, remove it from R
            state.R.discard(var)


def verify_register_pressure(spill_lines: List[str], k: int, file_name: str) -> Tuple[bool, List[RegPressureError], List[str]]:
    """
    Verify register pressure by simulating R and S sets through the program.
    Stops at the first error found.
    
    Args:
        spill_lines: List of lines from the SPILL section
        k: Number of available registers
        file_name: Name of the file being checked (for error messages)
        
    Returns:
        Tuple of (success, list of errors, annotated IR lines)
    """
    errors: List[RegPressureError] = []
    
    # Parse the original IR file to get liveness information
    original_function = None
    try:
        with open(file_name, "r") as f:
            ir_content = f.read()
        # Filter out CHECK directives and comments
        lines = ir_content.splitlines()
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip comment lines (starting with # or ;) and CHECK directives
            if (stripped.startswith("#") or 
                stripped.startswith(";") or 
                re.match(r'\s*;\s*CHECK-', line)):
                continue
            filtered_lines.append(line)
        ir_text = "\n".join(filtered_lines)
        original_function = parse_function(ir_text)
        liveness.compute_liveness(original_function)
    except Exception:
        # If we can't parse the original IR, continue without liveness checking
        # (fallback to original behavior)
        pass
    
    # Parse the SPILL IR
    try:
        function = parse_spill_ir(spill_lines)
    except Exception as e:
        errors.append(RegPressureError(
            error_type="PARSE_ERROR",
            message=f"Failed to parse SPILL IR: {e}",
            block_name="",
            line_number=0
        ))
        return False, errors, spill_lines
    
    if not function.blocks:
        errors.append(RegPressureError(
            error_type="NO_BLOCKS",
            message="No blocks found in SPILL IR",
            block_name="",
            line_number=0
        ))
        return False, errors, spill_lines
    
    # Track final state of each block
    block_final_states: Dict[str, RegisterState] = {}
    
    # Track active eviction events across blocks (persist until resolved)
    active_events: List[EvictionEvent] = []
    
    # Process blocks in topological order
    block_order = topological_order_spill_ir(function)
    
    for block_name in block_order:
        block = function.blocks[block_name]
        
        # Initialize register state for this block entry
        state = RegisterState()
        
        if not block.predecessors:
            # Entry block: start with empty registers and spill slots
            state.R = set()
            state.S = set()
        else:
            # Non-entry block: merge states from all predecessors
            # S_entry = intersection of all predecessor S sets (only reliably spilled on ALL paths)
            pred_states = [block_final_states[pred] for pred in block.predecessors if pred in block_final_states]
            
            if pred_states:
                # S is intersection: variable must be spilled on ALL paths
                state.S = set.intersection(*[s.S for s in pred_states]) if pred_states else set()
                
                # R is intersection: variable must be in register on ALL paths
                # But we need to be careful - if a variable is in R in one pred but not another,
                # it's not reliably in R at entry
                state.R = set.intersection(*[s.R for s in pred_states]) if pred_states else set()
            else:
                # No predecessor states available (shouldn't happen, but handle gracefully)
                state.R = set()
                state.S = set()
        
        # Remove dead variables from R at block entry
        # This is important because variables from predecessors might be dead in this block
        remove_dead_variables_from_r(state, block_name, 0, original_function, function)
        
        # Filter active events at block entry: update candidates based on current state
        # At merge points, candidates that are not in R ∩ S were evicted in at least one path
        # We keep tracking them until we see them used or reloaded to determine which was evicted
        for event in active_events[:]:
            # Update candidates: only keep those still possible (in R ∩ S at entry)
            # If a candidate is not in R ∩ S, it was evicted or used/spilled in all paths
            event.candidates &= (state.R & state.S)
            # If all candidates were removed, the event is resolved (all were evicted)
            if not event.candidates:
                active_events.remove(event)
        
        # Process each instruction in the block
        found_error = False
        for instr_idx, instr in enumerate(block.instructions):
            if found_error:
                break
                
            if instr.kind == "spill":
                var = instr.variable
                if var is None:
                    continue
                
                # Check if spilling this variable resolves an eviction event
                # (if this var was a candidate, spilling it confirms it wasn't evicted earlier)
                for event in active_events[:]:
                    if var in event.candidates:
                        # This var was spilled, so it wasn't the evicted one - remove from candidates
                        event.candidates.discard(var)
                        # If all candidates were resolved (spilled or reloaded), remove event
                        if not event.candidates:
                            active_events.remove(event)
                
                # Error: Spilling variable not in registers
                if var not in state.R:
                    errors.append(RegPressureError(
                        error_type="SPILL_NOT_IN_REGISTER",
                        message=f"spill {var} but {var} is not in registers (R={sorted(state.R)}, S={sorted(state.S)})",
                        block_name=block_name,
                        line_number=instr.line_number,
                        variable=var
                    ))
                    found_error = True
                    break
                
                # Error: Double-spill (variable already in S)
                if var in state.S:
                    errors.append(RegPressureError(
                        error_type="DOUBLE_SPILL",
                        message=f"spill {var} but {var} is already spilled (R={sorted(state.R)}, S={sorted(state.S)})",
                        block_name=block_name,
                        line_number=instr.line_number,
                        variable=var
                    ))
                    found_error = True
                    break
                
                # Perform spill: remove from R, add to S
                state.R.discard(var)
                state.S.add(var)
                
            elif instr.kind == "reload":
                var = instr.variable
                if var is None:
                    continue
                
                # Error: Reloading variable not in S (not spilled on all paths)
                if var not in state.S:
                    errors.append(RegPressureError(
                        error_type="RELOAD_WITHOUT_SPILL",
                        message=f"reload {var} but {var} is not in spill slots (R={sorted(state.R)}, S={sorted(state.S)})",
                        block_name=block_name,
                        line_number=instr.line_number,
                        variable=var
                    ))
                    found_error = True
                    break
                
                # Check if reloading this variable resolves an eviction event
                # (if this var was a candidate in an event, it confirms it was the evicted one)
                event_to_remove = None
                for event in active_events:
                    if var in event.candidates:
                        # This confirms var was the evicted one - resolve the event
                        event_to_remove = event
                        # Remove var from R (it was optimistically kept in R, but was actually evicted)
                        state.R.discard(var)
                        break
                
                if event_to_remove:
                    active_events.remove(event_to_remove)
                
                # Error: Double-reload (variable already in R, and not resolving an event)
                if var in state.R:
                    errors.append(RegPressureError(
                        error_type="DOUBLE_RELOAD",
                        message=f"reload {var} but {var} is already in registers (R={sorted(state.R)}, S={sorted(state.S)})",
                        block_name=block_name,
                        line_number=instr.line_number,
                        variable=var
                    ))
                    found_error = True
                    break
                
                # Check register pressure: would reload exceed k?
                # First, remove any dead variables from R
                # Check liveness just before this instruction (after previous instructions have been processed)
                check_idx = max(0, instr_idx - 1) if instr_idx > 0 else 0
                remove_dead_variables_from_r(state, block_name, check_idx, original_function, function)
                
                if len(state.R) >= k:
                    # Compute eviction candidates: variables in R that are also in S
                    eviction_candidates = state.R & state.S
                    
                    if not eviction_candidates:
                        # No silently-evictable candidates - immediate error
                        errors.append(RegPressureError(
                            error_type="REGISTER_PRESSURE_EXCEEDED",
                            message=f"reload {var} would exceed register limit k={k}, no silently-evictable candidates (R={sorted(state.R)}, S={sorted(state.S)}, |R|={len(state.R)})",
                            block_name=block_name,
                            line_number=instr.line_number,
                            variable=var
                        ))
                        found_error = True
                        break
                    else:
                        # Create eviction event: track which variables could have been evicted
                        event = EvictionEvent(
                            candidates=eviction_candidates.copy(),
                            reload_var=var,
                            line_number=instr.line_number,
                            block_name=block_name
                        )
                        active_events.append(event)
                        # Optimistically assume all candidates still in R (we'll track usage)
                        # Add reloaded var to R
                        state.R.add(var)
                else:
                    # No register pressure - just add to R
                    state.R.add(var)
                
            elif instr.kind == "op":
                # Before processing uses, remove any dead variables from R
                # This ensures we have accurate register pressure before checking uses
                remove_dead_variables_from_r(state, block_name, instr_idx, original_function, function)
                
                # Check uses: all used variables must be in registers
                for use_var in instr.uses:
                    # Check if this use affects any active eviction events
                    for event in active_events:
                        if use_var in event.candidates:
                            # Remove from candidates (this var was used, so it wasn't evicted)
                            event.candidates.discard(use_var)
                            # If all candidates were used, error: one must have been evicted
                            if not event.candidates:
                                errors.append(RegPressureError(
                                    error_type="USE_WITHOUT_REGISTER",
                                    message=f"op uses {use_var} but all eviction candidates from reload {event.reload_var} at {event.block_name}:{event.line_number} were used - one must have been evicted but was used without reload (R={sorted(state.R)}, S={sorted(state.S)})",
                                    block_name=block_name,
                                    line_number=instr.line_number,
                                    variable=use_var
                                ))
                                found_error = True
                                break
                    
                    if found_error:
                        break
                    
                    # Check if variable is in registers (after accounting for eviction events)
                    if use_var not in state.R:
                        errors.append(RegPressureError(
                            error_type="USE_WITHOUT_REGISTER",
                            message=f"op uses {use_var} but {use_var} is not in registers (R={sorted(state.R)}, S={sorted(state.S)})",
                            block_name=block_name,
                            line_number=instr.line_number,
                            variable=use_var
                        ))
                        found_error = True
                        break
                
                if found_error:
                    break
                
                # After processing uses, remove any dead variables from R
                # Variables that were just used and are no longer live should be removed
                remove_dead_variables_from_r(state, block_name, instr_idx + 1, original_function, function)
                
                # Process defs: add to registers
                for def_var in instr.defs:
                    # Check if defining this variable resolves an eviction event
                    # (if this var was a candidate, defining it confirms it wasn't evicted)
                    for event in active_events[:]:
                        if def_var in event.candidates:
                            # This var is being redefined, so it wasn't evicted - remove from candidates
                            event.candidates.discard(def_var)
                            # If all candidates were resolved, remove event
                            if not event.candidates:
                                active_events.remove(event)
                    
                    # Check register pressure before adding
                    # First, remove any dead variables from R
                    remove_dead_variables_from_r(state, block_name, instr_idx, original_function, function)
                    
                    if def_var not in state.R and len(state.R) >= k:
                        # Need to spill something, but this is an error condition
                        # (the algorithm should have spilled before this point)
                        errors.append(RegPressureError(
                            error_type="REGISTER_PRESSURE_EXCEEDED",
                            message=f"op defs {def_var} would exceed register limit k={k} (R={sorted(state.R)}, S={sorted(state.S)}, |R|={len(state.R)})",
                            block_name=block_name,
                            line_number=instr.line_number,
                            variable=def_var
                        ))
                        found_error = True
                        break
                    
                    # Add to registers
                    state.R.add(def_var)
                    # Note: S only shrinks at block entry merges, not when variables are defined
                
            elif instr.kind == "phi":
                # Phi defines a variable - similar to op defs
                for def_var in instr.defs:
                    # Check if defining this variable resolves an eviction event
                    for event in active_events[:]:
                        if def_var in event.candidates:
                            # This var is being redefined, so it wasn't evicted - remove from candidates
                            event.candidates.discard(def_var)
                            # If all candidates were resolved, remove event
                            if not event.candidates:
                                active_events.remove(event)
                    
                    # Check register pressure before adding
                    # First, remove any dead variables from R
                    remove_dead_variables_from_r(state, block_name, instr_idx, original_function, function)
                    
                    if def_var not in state.R and len(state.R) >= k:
                        errors.append(RegPressureError(
                            error_type="REGISTER_PRESSURE_EXCEEDED",
                            message=f"phi defs {def_var} would exceed register limit k={k} (R={sorted(state.R)}, S={sorted(state.S)}, |R|={len(state.R)})",
                            block_name=block_name,
                            line_number=instr.line_number,
                            variable=def_var
                        ))
                        found_error = True
                        break
                    
                    state.R.add(def_var)
                    # Note: S only shrinks at block entry merges, not when variables are defined
        
        # If we found an error, stop processing blocks
        if found_error:
            break
        
        # Store final state for this block
        block_final_states[block_name] = RegisterState(
            R=state.R.copy(),
            S=state.S.copy()
        )
    
    return len(errors) == 0, errors, spill_lines


def print_ir_with_errors(spill_lines: List[str], errors: List[RegPressureError]) -> None:
    """
    Print the IR with errors annotated inline where they occurred.
    
    Args:
        spill_lines: List of lines from the SPILL section
        errors: List of errors found (should have at least one)
    """
    if not errors:
        return
    
    # Create mapping of line number to error
    error_by_line: Dict[int, RegPressureError] = {}
    for error in errors:
        if error.line_number > 0:
            error_by_line[error.line_number] = error
    
    # Print the IR with error annotations
    for line_num, line in enumerate(spill_lines, start=1):
        if line_num in error_by_line:
            error = error_by_line[line_num]
            # Print the line with error annotation
            stripped = line.rstrip()
            # Add error annotation as a comment
            print(f"{stripped}  # ERROR: {error.error_type}: {error.message}", file=sys.stderr)
        else:
            print(line, file=sys.stderr)


def run_main_program(ir_file: str, k: int) -> str:
    """Run main.py and capture its output."""
    try:
        result = subprocess.run(
            [sys.executable, "main.py", ir_file, "-k", str(k)],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"{ir_file}: Error running main.py: {e}", file=sys.stderr)
        print(f"{ir_file}: stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="FileCheck-like tool for IR register allocation output"
    )
    parser.add_argument("file", help="Path to the IR file")
    parser.add_argument("-k", "--registers", type=int, default=3,
                        help="Number of available registers (default: 3)")
    parser.add_argument("--update", action="store_true",
                        help="Update CHECK directives in IR file based on current output")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Verify output against CHECK directives (default)")
    parser.add_argument("--verify-reg-pressure", action="store_true",default=True,
                        help="Verify register pressure by simulating R/S state through the program")
    
    args = parser.parse_args()
    
    # Read IR file
    try:
        with open(args.file, "r") as f:
            ir_content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Run main.py to get output
    output = run_main_program(args.file, args.registers)
    
    # Parse output sections
    sections = parse_output_sections(output)
    
    # Handle register pressure verification mode
    if args.verify_reg_pressure:
        spill_lines = sections.get(CheckType.SPILL, [])
        if not spill_lines:
            print(f"{args.file}: Error: No SPILL section found in output", file=sys.stderr)
            sys.exit(1)
        
        success, errors, annotated_lines = verify_register_pressure(spill_lines, args.registers, args.file)
        
        if success:
            print(f"{args.file}: Register pressure verification passed!")
            sys.exit(0)
        else:
            print(f"{args.file}: Register pressure verification failed:", file=sys.stderr)
            print("", file=sys.stderr)
            print("IR with errors:", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            print_ir_with_errors(annotated_lines, errors)
            sys.exit(1)
    
    if args.update:
        # Update mode: generate CHECK directives
        updated_content = generate_checks(sections, ir_content, args.registers)
        
        # Write back to file
        with open(args.file, "w") as f:
            f.write(updated_content)
        
        print(f"Updated CHECK directives in {args.file}")
    else:
        # Verify mode: check against existing directives
        checks = parse_checks(ir_content)
        
        # Filter checks to only those matching k (or with no k specified)
        # Count total checks after filtering
        total_checks = 0
        for check_type, directives in checks.items():
            filtered = [d for d in directives if d.k is None or d.k == args.registers]
            total_checks += len(filtered)
        
        if total_checks == 0:
            print(f"{args.file}: Warning: No CHECK directives found in IR file for k={args.registers}", file=sys.stderr)
            print(f"{args.file}: Use --update to generate CHECK directives", file=sys.stderr)
            sys.exit(1)
        
        success, errors, warnings = verify_checks(checks, sections, args.registers, args.file)
        
        # Display warnings about missing checks
        if warnings:
            print(f"{args.file}: Warning: Missing CHECK directives detected:", file=sys.stderr)
            for warning in warnings:
                print(f"  {warning}", file=sys.stderr)
            print("", file=sys.stderr)
        
        if success:
            if warnings:
                print(f"{args.file}: All CHECK directives passed, but some patterns in output lack CHECK directives.")
                sys.exit(0)
            else:
                print(f"{args.file}: All CHECK directives passed!")
                sys.exit(0)
        else:
            print(f"{args.file}: CHECK directive verification failed:", file=sys.stderr)
            for error in errors:
                print(f"  {error}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

