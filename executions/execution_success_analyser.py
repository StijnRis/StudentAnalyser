import ast
from difflib import ndiff
from typing import Dict

import pandas as pd

from enums import LearningGoal


def add_execution_successes_df(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add a DataFrame of execution successes to the data dictionary.
    """

    executions_df = data["executions"]

    successes = executions_df[executions_df["success"] == True]
    execution_successes_df = pd.DataFrame(
        {"id": successes.index, "execution_id": successes["id"]}
    ).reset_index(drop=True)
    data["execution_successes"] = execution_successes_df


def add_prev_successful_executed_file_version_id(data: Dict[str, pd.DataFrame]) -> None:
    """
    For each execution, find the id of the file version from the previous successful execution
    by the same user for the same file, and add it as 'prev_successful_executed_file_version_id'.
    """

    execution_successes_df = data["execution_successes"]
    executions_df = data["executions"]

    # Make table with relevant data
    merged = execution_successes_df.merge(
        executions_df[["id", "username", "file", "datetime", "file_version_id"]],
        left_on="execution_id",
        right_on="id",
        how="left",
    )
    merged = merged.sort_values(["username", "file", "datetime"]).reset_index(drop=True)

    # For each execution, find the previous successful execution
    def find_prev_successful_id(row):
        user = row["username"]
        file = row["file"]
        datetime = row["datetime"]
        prev_success = merged[
            (merged["username"] == user)
            & (merged["file"] == file)
            & (merged["datetime"] < datetime)
        ]
        if not prev_success.empty:
            return prev_success.iloc[-1]["file_version_id"]
        else:
            return None

    merged["prev_successful_executed_file_version_id"] = merged.apply(
        find_prev_successful_id, axis=1
    )

    # Merge the prev_successful_executed_file_version_id back into the original execution_successes_df
    execution_successes_df = execution_successes_df.merge(
        merged[["execution_id", "prev_successful_executed_file_version_id"]],
        on="execution_id",
        how="left",
    )
    data["execution_successes"] = execution_successes_df


def add_line_numbers_of_new_code(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add the line numbers of lines that were added in the current file version compared to the previous one.
    """

    file_versions_df = data["file_versions"]
    execution_successes_df = data["execution_successes"]
    executions_df = data["executions"]

    # Merge execution_successes with executions to get file_version_id
    merged = execution_successes_df.merge(
        executions_df[["id", "file_version_id"]],
        left_on="execution_id",
        right_on="id",
        how="left",
        suffixes=("", "_file_version"),
    )
    # Merge with file_versions to get code for current file version
    merged = merged.merge(
        file_versions_df[["id", "code"]],
        left_on="file_version_id",
        right_on="id",
        how="left",
        suffixes=("", "_file_version"),
    )
    # Merge with file_versions again to get previous code
    merged = merged.merge(
        file_versions_df[["id", "code"]].rename(
            columns={"id": "prev_id", "code": "prev_code"}
        ),
        left_on="prev_successful_executed_file_version_id",
        right_on="prev_id",
        how="left",
    )

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

    def compute_added_lines(row):
        prev_code = row["prev_code"] if pd.notnull(row["prev_code"]) else ""
        curr_code = row["code"]
        return get_line_numbers_of_added_code(prev_code, curr_code)

    # Compute added lines for each execution
    execution_successes_df["line_numbers_of_new_code"] = merged.apply(
        compute_added_lines, axis=1
    )

    # Store result
    data["execution_successes"] = execution_successes_df


def add_constructs_of_added_code(data: Dict[str, pd.DataFrame]) -> None:
    """
    Find the AST nodes (constructs) corresponding to the added lines of code.
    """

    file_versions_df = data["file_versions"]
    execution_successes_df = data["execution_successes"]
    executions_df = data["executions"]

    # Merge execution_successes with executions to get file_version_id
    merged = execution_successes_df.merge(
        executions_df[["id", "file_version_id"]],
        left_on="execution_id",
        right_on="id",
        how="left",
        suffixes=("", "_file_version"),
    )
    # Merge with file_versions to get code for current file version
    merged = merged.merge(
        file_versions_df[["id", "code"]],
        left_on="file_version_id",
        right_on="id",
        how="left",
        suffixes=("", "_file_version"),
    )
    # Merge with file_versions again to get previous code
    merged = merged.merge(
        file_versions_df[["id", "code"]].rename(
            columns={"id": "prev_id", "code": "prev_code"}
        ),
        left_on="prev_successful_executed_file_version_id",
        right_on="prev_id",
        how="left",
    )
        
    def get_ast_nodes_for_lines(row):
        code = row["code"]
        lines = row["line_numbers_of_new_code"]
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


    # Compute added constructs for each execution
    execution_successes_df["added_constructs"] = merged.apply(
        get_ast_nodes_for_lines, axis=1
    ).astype(object)
    data["execution_successes"] = execution_successes_df


def add_learning_goals_of_added_code(learning_goals: list[LearningGoal]):
    def add_learning_goals_of_added_code(data: Dict[str, pd.DataFrame]) -> None:
        """
        For each execution success, check for each learning goal if it is applied in one of the added constructs.
        Adds a column 'learning_goals_of_added_code' with a list of matching learning goal names.
        """

        execution_successes_df = data["execution_successes"]

        def detect_learning_goals(row):
            constructs = row["added_constructs"]
            matched_goals = []
            for construct in constructs:
                for goal in learning_goals:
                    if goal.is_applied(construct):
                        matched_goals.append(goal)
            return matched_goals

        execution_successes_df["learning_goals_of_added_code"] = (
            execution_successes_df.apply(detect_learning_goals, axis=1)
        )
        data["execution_successes"] = execution_successes_df

    return add_learning_goals_of_added_code
