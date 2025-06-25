import ast
import difflib

from enums import LearningGoal


def get_ranges_of_changed_code(old_code: str, new_code: str) -> list[tuple[int, int]]:
    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()
    changed_positions = []

    sm = difflib.SequenceMatcher(None, old_lines, new_lines)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        for new_line_index in range(j1, j2):
            new_line = new_lines[new_line_index]
            old_line_index = (
                i1 + (new_line_index - j1)
                if i1 + (new_line_index - j1) < len(old_lines)
                else None
            )
            old_line = old_lines[old_line_index] if old_line_index is not None else ""

            # Compare characters in the affected lines
            char_sm = difflib.SequenceMatcher(None, old_line, new_line)
            for ctag, ci1, ci2, cj1, cj2 in char_sm.get_opcodes():
                if ctag != "equal":
                    changed_positions.append((new_line_index + 1, cj1 + 1, cj2 + 1))

    return changed_positions


def get_ast_nodes_for_ranges(
    code: str, ranges: list[tuple[int, int, int]]
) -> list[ast.AST]:
    """
    Get AST nodes for specific line ranges in the code.
    :param code: The source code as a string.
    :param ranges: A list of tuples, each containing (line_number, start_char, end_char).
    """
    try:
        parsed_ast = ast.parse(code)
    except Exception:
        return []

    nodes = []

    possible_ranges = set((line, start, end) for line, start, end in ranges)

    def bfs(node):
        """
        Perform a breadth-first search to find nodes that fall within the possible ranges.
        When node fully covers a range, this range is removed from possible_ranges.
        """

        for child in ast.iter_child_nodes(node):
            bfs(child)

        if hasattr(node, "lineno"):
            if not hasattr(node, "col_offset"):
                raise ValueError(
                    "Node does not have col_offset attribute, which is required for range detection."
                )

            for line, column_start, column_end in possible_ranges:
                if line == node.lineno:
                    # Check if the node's column range overlaps with the given range
                    node_start = node.col_offset + 1  # Convert to 1-based index
                    node_end = node.end_col_offset + 1  # Convert to 1-based index
                    if not (node_end <= column_start or node_start >= column_end):
                        nodes.append(node)

                        # If the node fully covers the range, remove it from possible_ranges
                        if node_start <= column_start and node_end >= column_end:
                            possible_ranges.discard((line, column_start, column_end))

                        return

    bfs(parsed_ast)

    return nodes


def detect_learning_goals(
    constructs: list[ast.AST], learning_goals: list[LearningGoal]
) -> list[LearningGoal]:
    matched_goals = []
    for construct in constructs:
        for goal in learning_goals:
            if goal.is_applied(construct):
                matched_goals.append(goal)
    return matched_goals
