from typing import Dict

import pandas as pd


def add_execution_success(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add a column 'execution_success' to the executions DataFrame indicating whether the execution was successful.
    """

    executions_df = data["executions"]
    errors_df = data["execution_errors"]

    error_ids = set(errors_df["execution_id"])

    executions_df["success"] = ~executions_df["execution_id"].isin(error_ids)


def add_file_version_id(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add the id of the file version that was executed
    """

    executions_df = data["executions"]
    file_versions_df = data["file_versions"]

    # Merge with file versions on user_id, time, and file
    executions_df = executions_df.merge(
        file_versions_df[
            [
                "user_id",
                "datetime",
                "filename",
                "file_version_id",
            ]
        ],
        left_on=[
            "user_id",
            "datetime",
            "filename",
        ],
        right_on=[
            "user_id",
            "datetime",
            "filename",
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
        right_on="file_version_id",
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
                left_on="execution_id",
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
    executions_df = executions_df.sort_values(
        ["user_id", "filename", "datetime"]
    ).reset_index(drop=True)

    # Prepare columns
    executions_df["previous_success_id"] = None
    executions_df["previous_success_file_version_id"] = None
    executions_df["next_success_id"] = None
    executions_df["next_success_file_version_id"] = None
    executions_df["previous_error_id"] = None
    executions_df["previous_error_file_version_id"] = None
    executions_df["next_error_id"] = None
    executions_df["next_error_file_version_id"] = None
    executions_df["is_previous_execution_success"] = False

    for (user_id, filename), group in executions_df.groupby(["user_id", "filename"]):
        group = group.sort_values("datetime").reset_index()

        # Previous/next successful execution
        success_mask = group["success"]
        error_mask = ~group["success"]

        # Previous success
        group["previous_success_id"] = (
            group["execution_id"].where(success_mask).ffill().shift(1)
        )
        group["previous_success_file_version_id"] = (
            group["file_version_id"].where(success_mask).ffill().shift(1)
        )
        # Next success
        group["next_success_id"] = (
            group["execution_id"].where(success_mask).bfill().shift(-1)
        )
        group["next_success_file_version_id"] = (
            group["file_version_id"].where(success_mask).bfill().shift(-1)
        )
        # Previous error
        group["previous_error_id"] = (
            group["execution_id"].where(error_mask).ffill().shift(1)
        )
        group["previous_error_file_version_id"] = (
            group["file_version_id"].where(error_mask).ffill().shift(1)
        )
        # Next error
        group["next_error_id"] = (
            group["execution_id"].where(error_mask).bfill().shift(-1)
        )
        group["next_error_file_version_id"] = (
            group["file_version_id"].where(error_mask).bfill().shift(-1)
        )
        # Previous execution
        group["is_previous_execution_success"] = (
            group["success"].shift(1).fillna(False).astype(bool)
        )
        # Assign back
        executions_df.loc[
            group["index"],
            [
                "previous_success_id",
                "previous_success_file_version_id",
                "next_success_id",
                "next_success_file_version_id",
                "previous_error_id",
                "previous_error_file_version_id",
                "next_error_id",
                "next_error_file_version_id",
                "is_previous_execution_success",
            ],
        ] = group[
            [
                "previous_success_id",
                "previous_success_file_version_id",
                "next_success_id",
                "next_success_file_version_id",
                "previous_error_id",
                "previous_error_file_version_id",
                "next_error_id",
                "next_error_file_version_id",
                "is_previous_execution_success",
            ]
        ].values

    # Ensure ID columns keep their integer type (nullable Int64)
    for col in [
        "previous_success_id",
        "previous_success_file_version_id",
        "next_success_id",
        "next_success_file_version_id",
        "previous_error_id",
        "previous_error_file_version_id",
        "next_error_id",
        "next_error_file_version_id",
    ]:
        executions_df[col] = executions_df[col]

    data["executions"] = executions_df
