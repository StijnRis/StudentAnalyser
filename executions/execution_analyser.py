from typing import Dict

import pandas as pd


def add_execution_success(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add a column 'execution_success' to the executions DataFrame indicating whether the execution was successful.
    """

    executions_df = data["executions"]
    errors_df = data["execution_errors"]

    error_ids = set(errors_df["execution_id"])

    executions_df["success"] = ~executions_df["id"].isin(error_ids)


def add_file_version_id(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add the id of the file version that was executed
    """

    executions_df = data["executions"]
    file_versions_df = data["file_versions"]

    # Merge with file versions on username, time, and file
    executions_df = executions_df.merge(
        file_versions_df.rename(columns={"id": "file_version_id"})[
            ["username", "datetime", "file", "file_version_id"]
        ],
        left_on=["username", "datetime", "file"],
        right_on=["username", "datetime", "file"],
        how="left",
    )

    data["executions"] = executions_df


def add_id_of_previous_executed_file_version(
    data: Dict[str, pd.DataFrame],
) -> None:
    """
    Add the id of the file version that was previously executed by the same user for the same file.
    """
    
    executions_df = data["executions"]

    # Sort by username, file, and time to ensure correct order
    sorted_executions_df = executions_df.sort_values(
        ["username", "file", "datetime"]
    ).reset_index(drop=True)

    # Use apply to efficiently find the previous file version id for each execution
    def find_prev_id(row):
        user = row["username"]
        file = row["file"]
        datetime = row["datetime"]
        prev = sorted_executions_df[
            (sorted_executions_df["username"] == user)
            & (sorted_executions_df["file"] == file)
            & (sorted_executions_df["datetime"] < datetime)
        ]
        if not prev.empty:
            return prev.iloc[-1]["file_version_id"]
        else:
            return None

    executions_df["prev_executed_file_version_id"] = executions_df.apply(
        find_prev_id, axis=1
    )
    data["executions"] = executions_df
