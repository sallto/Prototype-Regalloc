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
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


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
        
        if success:
            # Only exit silently on success - no output
            sys.exit(0)
        else:
            # Only print when checks fail
            # Display warnings about missing checks (only if there are also errors)
            if warnings:
                print(f"{args.file}: Warning: Missing CHECK directives detected:", file=sys.stderr)
                for warning in warnings:
                    print(f"  {warning}", file=sys.stderr)
                print("", file=sys.stderr)
            
            print(f"{args.file}: CHECK directive verification failed:", file=sys.stderr)
            for error in errors:
                print(f"  {error}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

