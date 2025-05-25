import ast
from difflib import ndiff

from enums import LearningGoal


def get_line_numbers_of_added_code(old_code: str, new_code: str) -> list[int]:
    diff = ndiff((old_code).splitlines(), (new_code).splitlines())
    original_line = 0
    new_line = 0
    changes = []
    for line in diff:
        code = line[:2]
        if code == "  ":
            original_line += 1
            new_line += 1
        elif code == "- ":
            original_line += 1
        elif code == "+ ":
            changes.append(new_line + 1)
            new_line += 1
    return changes


def get_ast_nodes_for_lines(code: str, lines: list[int]) -> list[ast.AST]:
    try:
        parsed_ast = ast.parse(code)
    except Exception:
        return []

    nodes = []

    def dfs(node):
        if hasattr(node, "lineno") and getattr(node, "lineno", None) in lines:
            nodes.append(node)
        for child in ast.iter_child_nodes(node):
            dfs(child)

    dfs(parsed_ast)

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
