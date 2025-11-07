from typing import Dict

import pandas as pd


def add_timeline_df(data: Dict[str, pd.DataFrame]) -> None:
    """
    Create a timeline DataFrame by merging executions, execution_successes, execution_errors, and edits.
    The resulting DataFrame is stored as 'timeline' in the data dict.
    """

    # Prepare subsets of each dataframe
    executions_subset = (
        data["executions"]
        .merge(
            data["file_versions"],
            on=["file_version_id"],
            how="left",
            suffixes=("", "_file_version"),
        )[["execution_id", "user_id", "datetime", "filename", "code"]]
        .rename(columns={"execution_id": "event_id", "code": "value"})
    )
    executions_subset["key"] = "code"
    executions_subset["event_type"] = "execution"

    execution_outputs_subset = (
        data["execution_outputs"]
        .merge(
            data["executions"],
            on="execution_id",
            how="left",
        )[["execution_output_id", "user_id", "datetime", "filename", "output_text"]]
        .rename(columns={"execution_output_id": "event_id", "output_text": "value"})
    )
    execution_outputs_subset["key"] = "output_text"
    execution_outputs_subset["event_type"] = "execution_output"

    execution_successes_subset = (
        data["execution_successes"]
        .merge(
            data["executions"],
            on="execution_id",
            how="left",
        )[
            [
                "execution_success_id",
                "user_id",
                "datetime",
                "filename",
                "added_constructs_as_string",
            ]
        ]
        .rename(
            columns={
                "execution_success_id": "event_id",
                "added_constructs_as_string": "value",
            }
        )
    )
    execution_successes_subset["key"] = "added_constructs_as_string"
    execution_successes_subset["event_type"] = "execution_success"

    execution_errors_subset = data["execution_errors"].merge(
        data["executions"],
        on="execution_id",
        how="left",
    )
    execution_errors_subset["event_type"] = "execution_error"

    execution_errors_subset_1 = execution_errors_subset[
        [
            "execution_error_id",
            "user_id",
            "datetime",
            "filename",
            "traceback_no_formatting",
        ]
    ].rename(
        columns={
            "execution_error_id": "event_id",
            "traceback_no_formatting": "value",
        }
    )
    execution_errors_subset_1["key"] = "traceback_no_formatting"

    execution_errors_subset_2 = execution_errors_subset[
        [
            "execution_error_id",
            "user_id",
            "datetime",
            "filename",
            "changed_constructs_as_string_next_success",
        ]
    ].rename(
        columns={
            "execution_error_id": "event_id",
            "changed_constructs_as_string_next_success": "value",
        }
    )
    execution_errors_subset_2["key"] = "changed_constructs_as_string_next_success"

    messages_subset = data["messages"][
        ["message_id", "user_id", "datetime", "sender", "body"]
    ]
    messages_subset = messages_subset.rename(
        columns={"message_id": "event_id", "body": "value"}
    )
    messages_subset["key"] = messages_subset["user_id"].astype(str) + " message"
    messages_subset["event_type"] = "message"
    messages_subset = messages_subset.drop(columns=["sender"])

    # Concatenate all events into a single DataFrame
    timeline_df = pd.concat(
        [
            executions_subset,
            execution_outputs_subset,
            execution_successes_subset,
            execution_errors_subset_1,
            execution_errors_subset_2,
            messages_subset,
        ],
        ignore_index=True,
    )

    # Sort by datetime to create the timeline
    timeline_df = timeline_df.sort_values(by=["user_id", "datetime"]).reset_index(
        drop=True
    )
    timeline_df = timeline_df[
        ["event_id", "user_id", "datetime", "filename", "event_type", "key", "value"]
    ]

    data["timeline"] = timeline_df
