from enum import Enum
from typing import Dict

import pandas as pd

from executions.execution_error_analyser import ExecutionErrorCols
from executions.execution_cols import ExecutionsCols


def add_execution_success(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add a column 'execution_success' to the executions DataFrame indicating whether the execution was successful.
    """

    executions_df = data["executions"]
    errors_df = data["execution_errors"]

    error_ids = set(errors_df[ExecutionErrorCols.ID.value])

    executions_df[ExecutionsCols.SUCCESS.value] = ~executions_df[
        ExecutionsCols.ID.value
    ].isin(error_ids)


def add_file_version_id(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add the id of the file version that was executed
    """

    executions_df = data["executions"]
    file_versions_df = data["file_versions"]

    # Merge with file versions on username, time, and file
    executions_df = executions_df.merge(
        file_versions_df.rename(columns={"id": ExecutionsCols.FILE_VERSION_ID.value})[
            [
                ExecutionsCols.USERNAME.value,
                ExecutionsCols.DATETIME.value,
                ExecutionsCols.FILE.value,
                ExecutionsCols.FILE_VERSION_ID.value,
            ]
        ],
        left_on=[
            ExecutionsCols.USERNAME.value,
            ExecutionsCols.DATETIME.value,
            ExecutionsCols.FILE.value,
        ],
        right_on=[
            ExecutionsCols.USERNAME.value,
            ExecutionsCols.DATETIME.value,
            ExecutionsCols.FILE.value,
        ],
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
        [
            ExecutionsCols.USERNAME.value,
            ExecutionsCols.FILE.value,
            ExecutionsCols.DATETIME.value,
        ]
    ).reset_index(drop=True)

    # Use apply to efficiently find the previous file version id for each execution
    def find_prev_id(row):
        user = row[ExecutionsCols.USERNAME.value]
        file = row[ExecutionsCols.FILE.value]
        datetime = row[ExecutionsCols.DATETIME.value]
        prev = sorted_executions_df[
            (sorted_executions_df[ExecutionsCols.USERNAME.value] == user)
            & (sorted_executions_df[ExecutionsCols.FILE.value] == file)
            & (sorted_executions_df[ExecutionsCols.DATETIME.value] < datetime)
        ]
        if not prev.empty:
            return prev.iloc[-1][ExecutionsCols.FILE_VERSION_ID.value]
        else:
            return None

    executions_df[ExecutionsCols.PREV_EXECUTED_FILE_VERSION_ID] = [
        find_prev_id(row) for _, row in executions_df.iterrows()
    ]
    data["executions"] = executions_df


def add_execution_overview_df(data: Dict[str, pd.DataFrame]) -> None:
    """
    Merge all DataFrames that have an 'execution_id' column into a single overview DataFrame.
    Start with the 'executions' table (using 'id' as the key), then merge all others on 'execution_id',
    and finally merge the file_versions table on 'file_version_id' from executions.
    The resulting DataFrame is stored as 'execution_overview' in the data dict.
    """

    overview_df = data["executions"].copy()

    # Merge file_versions on file_version_id from executions
    file_versions_df = data["file_versions"]
    overview_df = overview_df.merge(
        file_versions_df,
        left_on="file_version_id",
        right_on="id",
        how="left",
        suffixes=(None, "_file_version"),
    )

    # Merge all DataFrames with 'execution_id' except executions itself
    for key, df in data.items():
        if key == "executions":
            continue
        if "execution_id" in df.columns:
            overview_df = overview_df.merge(
                df,
                left_on="id",
                right_on="execution_id",
                how="left",
                suffixes=(None, f"_{key}"),
            )

    data["execution_overview"] = overview_df


def add_surrounding_executions(data: Dict[str, pd.DataFrame]) -> None:
    """
    For each execution, find the previous and next successful and errored execution for the same user and file.
    Adds columns:
        - prev_success_id, next_success_id
        - prev_error_id, next_error_id
    """
    executions_df = data["executions"]
    # Sort by username, file, and datetime for efficient lookups
    executions_df = executions_df.sort_values(
        ["username", "file", ExecutionsCols.DATETIME.value]
    ).reset_index(drop=True)

    # Prepare output columns
    prev_success_ids = []
    next_success_ids = []
    prev_error_ids = []
    next_error_ids = []

    # Group by user and file for efficient lookups
    grouped = executions_df.groupby(["username", "file"], sort=False)

    for idx, row in executions_df.iterrows():
        user = row["username"]
        file = row["file"]
        dt = row[ExecutionsCols.DATETIME.value]
        group = grouped.get_group((user, file))

        # Previous successful execution
        prev_success = group[
            (group[ExecutionsCols.DATETIME.value] < dt)
            & (group[ExecutionsCols.SUCCESS.value])
        ].tail(1)
        prev_success_id = prev_success["id"].iloc[0] if not prev_success.empty else None
        prev_success_ids.append(prev_success_id)

        # Next successful execution
        next_success = group[
            (group[ExecutionsCols.DATETIME.value] > dt)
            & (group[ExecutionsCols.SUCCESS.value])
        ].head(1)
        next_success_id = next_success["id"].iloc[0] if not next_success.empty else None
        next_success_ids.append(next_success_id)

        # Previous errored execution
        prev_error = group[
            (group[ExecutionsCols.DATETIME.value] < dt)
            & (~group[ExecutionsCols.SUCCESS.value])
        ].tail(1)
        prev_error_id = prev_error["id"].iloc[0] if not prev_error.empty else None
        prev_error_ids.append(prev_error_id)

        # Next errored execution
        next_error = group[
            (group[ExecutionsCols.DATETIME.value] > dt)
            & (~group[ExecutionsCols.SUCCESS.value])
        ].head(1)
        next_error_id = next_error["id"].iloc[0] if not next_error.empty else None
        next_error_ids.append(next_error_id)

    executions_df["previous_success_id"] = prev_success_ids
    executions_df["next_success_id"] = next_success_ids
    executions_df["previous_error_id"] = prev_error_ids
    executions_df["next_error_id"] = next_error_ids
    data["executions"] = executions_df
