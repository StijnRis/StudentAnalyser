from typing import Dict

import pandas as pd

from enums import LearningGoal
from executions.execution_utils import (
    detect_learning_goals,
    get_ast_nodes_for_ranges,
    get_ranges_of_changed_code,
)


def add_execution_successes_df(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add a DataFrame of execution successes to the data dictionary.
    """

    executions_df = data["executions"]
    successes = executions_df[executions_df["success"] == True]
    execution_successes_df = pd.DataFrame(
        {
            "id": successes.index,
            "execution_id": successes["id"],
        }
    ).reset_index(drop=True)
    data["execution_successes"] = execution_successes_df


def add_new_code_analysis(learning_goals: list[LearningGoal]):
    def add_new_code_analysis(data: Dict[str, pd.DataFrame]) -> None:
        """
        For each execution success, compute:
            - line numbers of new code (added lines compared to previous success)
            - AST constructs of the new code
            - Learning goals applied in the new code
        """
        file_versions_df = data["file_versions"]
        execution_successes_df = data["execution_successes"]
        executions_df = data["executions"]

        # Merge execution_successes with executions
        merged = execution_successes_df.merge(
            executions_df[
                [
                    "id",
                    "file_version_id",
                    "previous_success_file_version_id",
                ]
            ],
            left_on="execution_id",
            right_on="id",
            how="left",
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
            file_versions_df[["id", "code"]],
            left_on="previous_success_file_version_id",
            right_on="id",
            how="left",
            suffixes=("", "_previous_version"),
        )

        def compute_line_numbers_of_new_code(row):
            code_previous_version = (
                row["code_previous_version"]
                if pd.notnull(row["code_previous_version"])
                else ""
            )
            code_current = row["code"]
            return get_ranges_of_changed_code(code_previous_version, code_current)

        def compute_added_constructs(row):
            code = row["code"]
            ranges = row["ranges_of_new_code"]
            return get_ast_nodes_for_ranges(code, ranges)

        def compute_learning_goals_of_added_code(row):
            constructs = row["added_constructs"]
            return detect_learning_goals(constructs, learning_goals)

        # Compute all columns in sequence
        merged["ranges_of_new_code"] = merged.apply(
            compute_line_numbers_of_new_code, axis=1
        )
        execution_successes_df["ranges_of_new_code"] = merged["ranges_of_new_code"]
        merged["added_constructs"] = merged.apply(
            compute_added_constructs, axis=1
        ).astype(object)
        execution_successes_df["added_constructs"] = merged["added_constructs"]
        merged["learning_goals_of_added_code"] = merged.apply(
            compute_learning_goals_of_added_code, axis=1
        )
        execution_successes_df["learning_goals_of_added_code"] = merged[
            "learning_goals_of_added_code"
        ]

        data["execution_successes"] = execution_successes_df

    return add_new_code_analysis
