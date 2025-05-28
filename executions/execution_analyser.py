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
            [
                "username",
                "datetime",
                "file",
                "file_version_id",
            ]
        ],
        left_on=[
            "username",
            "datetime",
            "file",
        ],
        right_on=[
            "username",
            "datetime",
            "file",
        ],
        how="left",
    )

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
        - previous_success_file_version_id, next_success_file_version_id
        - previous_error_file_version_id, next_error_file_version_id
    """
    executions_df = data["executions"]
    # Sort by username, file, and datetime for efficient lookups
    executions_df = executions_df.sort_values(
        ["username", "file", "datetime"]
    ).reset_index(drop=True)

    # Prepare output columns
    prev_success_ids = []
    prev_success_file_version_ids = []
    next_success_ids = []
    next_success_file_version_ids = []
    prev_error_ids = []
    prev_error_file_version_ids = []
    next_error_ids = []
    next_error_file_version_ids = []
    is_previous_execution_success = []

    # Group by user and file for efficient lookups
    grouped = executions_df.groupby(["username", "file"], sort=False)

    for idx, row in executions_df.iterrows():
        user = row["username"]
        file = row["file"]
        dt = row["datetime"]
        group = grouped.get_group((user, file))

        # Previous successful execution
        prev_success = group[(group["datetime"] < dt) & (group["success"])].tail(1)
        if not prev_success.empty:
            prev_success_id = prev_success["id"].iloc[0]
            prev_success_file_version_id = prev_success["file_version_id"].iloc[0]
        else:
            prev_success_id = None
            prev_success_file_version_id = None
        prev_success_ids.append(prev_success_id)
        prev_success_file_version_ids.append(prev_success_file_version_id)

        # Next successful execution
        next_success = group[(group["datetime"] > dt) & (group["success"])].head(1)
        if not next_success.empty:
            next_success_id = next_success["id"].iloc[0]
            next_success_file_version_id = next_success["file_version_id"].iloc[0]
        else:
            next_success_id = None
            next_success_file_version_id = None
        next_success_ids.append(next_success_id)
        next_success_file_version_ids.append(next_success_file_version_id)

        # Previous errored execution
        prev_error = group[(group["datetime"] < dt) & (~group["success"])].tail(1)
        if not prev_error.empty:
            prev_error_id = prev_error["id"].iloc[0]
            prev_error_file_version_id = prev_error["file_version_id"].iloc[0]
        else:
            prev_error_id = None
            prev_error_file_version_id = None
        prev_error_ids.append(prev_error_id)
        prev_error_file_version_ids.append(prev_error_file_version_id)

        # Next errored execution
        next_error = group[(group["datetime"] > dt) & (~group["success"])].head(1)
        if not next_error.empty:
            next_error_id = next_error["id"].iloc[0]
            next_error_file_version_id = next_error["file_version_id"].iloc[0]
        else:
            next_error_id = None
            next_error_file_version_id = None
        next_error_ids.append(next_error_id)
        next_error_file_version_ids.append(next_error_file_version_id)

        # Previous execution (any kind)
        prev_exec = group[group["datetime"] < dt].tail(1)
        if not prev_exec.empty:
            is_prev_success = bool(prev_exec["success"].iloc[0])
        else:
            is_prev_success = False
        is_previous_execution_success.append(is_prev_success)

    executions_df["previous_success_id"] = prev_success_ids
    executions_df["previous_success_file_version_id"] = prev_success_file_version_ids
    executions_df["next_success_id"] = next_success_ids
    executions_df["next_success_file_version_id"] = next_success_file_version_ids
    executions_df["previous_error_id"] = prev_error_ids
    executions_df["previous_error_file_version_id"] = prev_error_file_version_ids
    executions_df["next_error_id"] = next_error_ids
    executions_df["next_error_file_version_id"] = next_error_file_version_ids
    executions_df["is_previous_execution_success"] = is_previous_execution_success
    data["executions"] = executions_df
