import json
import os

import numpy as np
import pandas as pd

# ----------------------
# Helper Functions
# ----------------------


def _map_labels(df, labels_column, label_map):
    if label_map:
        label_map_lower = {str(k).lower(): str(v) for k, v in label_map.items()}
        for col in labels_column:
            df[col] = df[col].astype(str).str.lower().replace(label_map_lower)
        for col in labels_column:
            unique_vals = set(df[col].dropna().unique())
            unique_vals.discard("nan")
            unique_vals.discard(np.nan)
            unexpected = unique_vals - set(label_map_lower.values())
            if unexpected:
                raise ValueError(
                    f"Unexpected label(s) found in column '{col}': {unexpected}. "
                    f"Allowed values: {set(label_map_lower.values())}"
                )
    return df


def _normalize_column(df, src_col, norm_col):
    df[norm_col] = df[src_col].astype(str).str.strip().str.lower()
    return df


def _check_duplicates(df, column, labels_column):
    for name, group in df.groupby(column):
        for col in labels_column:
            if group[col].nunique() > 1:
                raise ValueError(
                    f"Duplicate entry '{name}' with different labels in column '{col}'"
                )
    df = df.drop_duplicates(subset=[column])
    return df

def _merge_and_update(
    target_df,
    target_merge_on,
    labels_df,
    labels_merge_on,
    labels_column,
    final_column_names,
):
    if labels_df[labels_merge_on].duplicated().any():
        raise ValueError(
            f"Column '{labels_merge_on}' in labels DataFrame must not contain duplicates."
        )
       
    merged = target_df.merge(
        labels_df[[labels_merge_on] + labels_column],
        left_on=target_merge_on,
        right_on=labels_merge_on,
        how="left",
    )
    merged = merged.set_index(target_df.index)
    for i, label in enumerate(labels_column):
        col = final_column_names[i]
        if col in target_df:
            mask = target_df[col].isna()
            target_df.loc[mask, col] = merged.loc[mask, label]
        else:
            target_df[col] = merged[label]

    # Unmatched warning logic
    unmatched = labels_df[~labels_df[labels_merge_on].isin(target_df[target_merge_on])]
    unmatched_count = len(unmatched)
    if unmatched_count > 0:
        print(
            f"Warning: {unmatched_count} records from the Excel file could not be matched."
        )
    return target_df


# ----------------------
# Loader Pipelines
# ----------------------


def generate_load_labelled_questions(
    base_data_path: str, metadata_for_analyser_file: str
):
    """Return the loader pipeline based on metadata JSON for labelled questions."""
    with open(metadata_for_analyser_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    pipeline = []
    for entry in metadata["LABELED_QUESTIONS"]:
        path = os.path.join(base_data_path, entry["path"])
        question_column = entry["question_column"]
        labels_column = entry["labels_column"]
        final_column_names = entry["final_column_names"]
        label_map = entry.get("label_map", None)
        pipeline.append(
            load_labelled_questions(
                path, question_column, labels_column, final_column_names, label_map
            )
        )
    return pipeline


def generate_load_labelled_traceback_errors(
    base_data_path: str, metadata_for_analyser_file: str
):
    """Return the loader pipeline based on metadata JSON for labelled traceback errors."""
    with open(metadata_for_analyser_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    pipeline = []
    for entry in metadata["LABELED_TRACEBACKS"]:
        path = os.path.join(base_data_path, entry["path"])
        traceback_column = entry["traceback_column"]
        labels_column = entry["labels_column"]
        final_column_names = entry["final_column_names"]
        label_map = entry.get("label_map", None)
        pipeline.append(
            load_labelled_traceback_errors(
                path, traceback_column, labels_column, final_column_names, label_map
            )
        )
    return pipeline


# ----------------------
# Loader Functions
# ----------------------


def load_labelled_questions(
    file_path: str,
    question_column: str,
    labels_column: list[str],
    final_column_names: list[str],
    label_map: dict[str, str] | None,
):
    def loader(data: dict[str, pd.DataFrame]):
        print(f"Loading xlsx file: {file_path}")
        df = pd.read_excel(file_path)
        df = _map_labels(df, labels_column, label_map)
        df = _normalize_column(df, question_column, "question_normalized")
        df = _check_duplicates(df, "question_normalized", labels_column)
        messages = data["messages"]
        interactions = data["interactions"]
        messages = _normalize_column(messages, "body", "body_normalized")
        merged = interactions.merge(
            messages[["message_id", "body", "body_normalized"]],
            left_on="question_id",
            right_on="message_id",
            how="left",
        )
        merged = _merge_and_update(
            merged,
            "body_normalized",
            df,
            "question_normalized",
            labels_column,
            final_column_names,
        )
        merged = merged.set_index(interactions.index)
        for i, label in enumerate(labels_column):
            col = final_column_names[i]
            if col in data["interactions"]:
                mask = data["interactions"][col].isna()
                data["interactions"].loc[mask, col] = merged.loc[mask, col]
            else:
                data["interactions"][col] = merged[col]

    return loader


def load_labelled_traceback_errors(
    file_path: str,
    traceback_column: str,
    labels_column: list[str],
    final_column_names: list[str],
    label_map: dict[str, str] | None,
):
    def loader(data: dict[str, pd.DataFrame]):
        print(f"Loading xlsx file: {file_path}")
        df = pd.read_excel(file_path)
        df = _map_labels(df, labels_column, label_map)
        df = _normalize_column(df, traceback_column, "error_normalized")
        df = _check_duplicates(df, "error_normalized", labels_column)
        execution_errors = data["execution_errors"]
        # Create a temporary normalized column for matching, but do not keep it in the result
        execution_errors = _normalize_column(
            execution_errors, "traceback_no_formatting", "traceback_normalized"
        )
        execution_errors = _merge_and_update(
            execution_errors,
            "traceback_normalized",
            df,
            "error_normalized",
            labels_column,
            final_column_names,
        )
        # Special handling for learning_goals_in_error: split by comma and strip whitespace, store as list
        for col in final_column_names:
            execution_errors[col] = execution_errors[col].apply(
                lambda x: (
                    [s.strip() for s in x.split(",")]
                    if isinstance(x, str) and x.strip()
                    else x
                )
            )
        # Remove the temporary normalized column
        execution_errors = execution_errors.drop(columns=["traceback_normalized"])
        data["execution_errors"] = execution_errors

    return loader
