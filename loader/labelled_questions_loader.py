import os

import pandas as pd

from enums import QuestionType

LABEL_MAP = {  # In comments are the original labels
    0: "Answer to question of chatbot",  # "Chatbot interaction"
    1: "Concept comprehension",
    2: "Copied question",  # "Copying Notebook Questions"
    3: "Code comprehension",
    4: "Fix code",  # "Fix this code / error"
    5: "Error comprehension",
    6: "Task delegation",  # "Task Related Delegation"
    7: "Pasted code without context",  # "Pasting Code Without Explanation",
    8: "Other",  # "Random"
    9: "Question comprehension",
}


def load_labelled_questions(
    file_path: str,
    question_column: str,
    labels_column: list[str]
):
    def load_labelled_questions(data: dict[str, pd.DataFrame]):
        """
        Loads a labelled questions file, matches each question to a message, verifies the interaction, and adds a label column efficiently.
        """
        print(f"Loading xlsx file: {file_path}")
        df = pd.read_excel(file_path)

        # Find corresponding message for each question
        messages = data["messages"]
        interactions = data["interactions"]

        # Merge on question text (assuming exact match)
        merged = df.merge(
            messages,
            left_on=question_column,
            right_on="body",
            how="left",
            suffixes=(None, "_msg"),
        )
        if merged["message_id"].isnull().any():
            unmatched_count = merged["message_id"].isnull().sum()
            print(f"Warning: {unmatched_count} questions could not be matched to a message.")
        
        # Find the interaction for each message
        merged = merged.merge(
            interactions,
            left_on="message_id",
            right_on="question_id",
            how="left",
            suffixes=(None, "_int"),
        )
        if merged["interaction_id"].isnull().any():
            raise RuntimeError(
                "Some messages could not be matched to an interaction as a question."
            )
        
        # Check that each matched message is a question (not an answer)
        if (merged["question_id"] != merged["message_id"]).any():
            raise RuntimeError(
                "A matched message is not the question in its interaction."
            )
        
        # Add a column to the messages DataFrame with the label, using file name and column name
        file_base = os.path.splitext(os.path.basename(file_path))[0]
        for col in labels_column:
            label_col = f"{file_base}_{col}"
            # Create a Series mapping message_id to label
            label_map_series = merged.set_index("message_id")[col]
            # Only assign once, vectorized
            data["messages"][label_col] = data["messages"]["message_id"].map(
                label_map_series
            )

    return load_labelled_questions
