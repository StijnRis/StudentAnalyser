import ast
from difflib import ndiff
from typing import Dict

import pandas as pd

from enums import LearningGoal
from executions.execution_success_cols import ExecutionSuccessCols
from executions.execution_analyser import ExecutionsCols
from file_versions.file_version_cols import FileVersionsCols


def add_execution_successes_df(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add a DataFrame of execution successes to the data dictionary.
    """

    executions_df = data["executions"]
    successes = executions_df[executions_df["success"] == True]
    execution_successes_df = pd.DataFrame(
        {
            ExecutionSuccessCols.ID.value: successes.index,
            ExecutionSuccessCols.EXECUTION_ID.value: successes["id"],
        }
    ).reset_index(drop=True)
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
        executions_df[
            [
                ExecutionsCols.ID.value,
                ExecutionsCols.FILE_VERSION_ID.value,
                ExecutionsCols.PREVIOUS_SUCCESS_ID.value,
            ]
        ],
        left_on=ExecutionSuccessCols.EXECUTION_ID.value,
        right_on=ExecutionSuccessCols.ID.value,
        how="left",
        suffixes=("", "_file_version"),
    )
    # Merge with file_versions to get code for current file version
    merged = merged.merge(
        file_versions_df[[FileVersionsCols.ID.value, FileVersionsCols.CODE.value]],
        left_on=ExecutionsCols.FILE_VERSION_ID.value,
        right_on=FileVersionsCols.ID.value,
        how="left",
        suffixes=("", "_file_version"),
    )
    # Merge with file_versions again to get previous code
    merged = merged.merge(
        file_versions_df,
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
        curr_code = row[FileVersionsCols.CODE.value]
        return get_line_numbers_of_added_code(prev_code, curr_code)

    # Compute added lines for each execution
    execution_successes_df[ExecutionSuccessCols.LINE_NUMBERS_OF_NEW_CODE.value] = (
        merged.apply(compute_added_lines, axis=1)
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
        executions_df[[ExecutionsCols.ID.value, ExecutionsCols.FILE_VERSION_ID.value]],
        left_on=ExecutionSuccessCols.EXECUTION_ID.value,
        right_on=ExecutionsCols.ID.value,
        how="left",
        suffixes=("", "_file_version"),
    )
    # Merge with file_versions to get code for current file version
    merged = merged.merge(
        file_versions_df[[FileVersionsCols.ID.value, FileVersionsCols.CODE.value]],
        left_on=ExecutionsCols.FILE_VERSION_ID.value,
        right_on=FileVersionsCols.ID.value,
        how="left",
        suffixes=("", "_file_version"),
    )
    # Merge with file_versions again to get previous code
    merged = merged.merge(
        file_versions_df[
            [FileVersionsCols.ID.value, FileVersionsCols.CODE.value]
        ].rename(
            columns={
                FileVersionsCols.CODE.value: "prev_code",
            }
        ),
        left_on="prev_successful_executed_file_version_id",
        right_on=ExecutionSuccessCols.ID.value,
        how="left",
    )

    def get_ast_nodes_for_lines(row):
        code = row[FileVersionsCols.CODE.value]
        lines = row[ExecutionSuccessCols.LINE_NUMBERS_OF_NEW_CODE.value]
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
    execution_successes_df[ExecutionSuccessCols.ADDED_CONSTRUCTS.value] = merged.apply(
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
            constructs = row[ExecutionSuccessCols.ADDED_CONSTRUCTS.value]
            matched_goals = []
            for construct in constructs:
                for goal in learning_goals:
                    if goal.is_applied(construct):
                        matched_goals.append(goal)
            return matched_goals

        execution_successes_df[
            ExecutionSuccessCols.LEARNING_GOALS_OF_ADDED_CODE.value
        ] = execution_successes_df.apply(detect_learning_goals, axis=1)
        data["execution_successes"] = execution_successes_df

    return add_learning_goals_of_added_code
