import json
import os

import numpy as np
import pandas as pd


def generate_load_questions_pipeline(
    base_data_path: str, metadata_for_analyser_file: str
):
    """Return the loader pipeline based on metadata JSON."""

    # Load metadata from JSON file
    with open(metadata_for_analyser_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Generate the pipeline
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


def load_labelled_questions(
    file_path: str,
    question_column: str,
    labels_column: list[str],
    final_column_names: list[str],
    label_map: dict[str, str] | None,
):
    def load_labelled_questions(data: dict[str, pd.DataFrame]):
        """
        Loads a labelled questions file, matches each question to a message, verifies the interaction, and adds a label column efficiently.
        """
        print(f"Loading xlsx file: {file_path}")
        df = pd.read_excel(file_path)

        # map labels
        if label_map:
            # Ensure all label columns are strings before mapping
            df[labels_column] = df[labels_column].astype(str).replace(label_map)
            # Check for unexpected values after mapping
            for col in labels_column:
                unique_vals = set(df[col].dropna().unique())
                # Remove string 'nan' and np.nan from unique values
                unique_vals.discard("nan")
                unique_vals.discard(np.nan)
                unexpected = unique_vals - set(label_map.values())
                if unexpected:
                    raise ValueError(
                        f"Unexpected label(s) found in column '{col}': {unexpected}. "
                        f"Allowed values: {set(label_map.values())}"
                    )

        # Find corresponding message for each question
        messages = data["messages"]
        interactions = data["interactions"]

        # Normalize question text for matching
        messages["body_normalized"] = (
            messages["body"].astype(str).str.strip().str.lower()
        )
        df["question_normalized"] = (
            df[question_column].astype(str).str.strip().str.lower()
        )

        # Check for same label of duplicate questions in the Excel file and remove
        for name, group in df.groupby("question_normalized"):
            for col in labels_column:
                if group[col].nunique() > 1:
                    raise ValueError(f"Duplicate question '{name}' with different labels in column '{col}'")
        df = df.drop_duplicates(subset=["question_normalized"] + labels_column)

        # Merge on normalized question text
        merged = interactions.merge(
            messages[["message_id", "body", "body_normalized"]],
            left_on="question_id",
            right_on="message_id",
            how="left",
        )
        merged = merged.merge(
            df[["question_normalized"] + labels_column],
            left_on="body_normalized",
            right_on="question_normalized",
            how="left",
        )

        # Calculate how many questions from the Excel file could not be matched to a message
        unmatched_questions = df[
            ~df["question_normalized"].isin(messages["body_normalized"])
        ]
        unmatched_count = len(unmatched_questions)
        if unmatched_count > 0:
            print(
                f"Warning: {unmatched_count} questions from the Excel file could not be matched to a message."
            )

        # Save results to interactions dataframe
        merged = merged.set_index(interactions.index)
        for i, label in enumerate(labels_column):
            col = final_column_names[i]
            if col in data["interactions"]:
                # Only update where current value is null/NaN
                mask = data["interactions"][col].isna()
                data["interactions"].loc[mask, col] = merged.loc[mask, label]
            else:
                data["interactions"][col] = merged[label]

    return load_labelled_questions
