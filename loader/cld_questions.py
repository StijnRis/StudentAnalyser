import csv

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


def load_cld_questions(file_path: str, question_types: list[QuestionType]):
    def load_cld_questions(data: dict[str, pd.DataFrame]):
        """
        Loads a CSV file containing CLD questions and their labels.
        Handles lines with multiple commas by splitting on the last comma.
        Skips the first line (header).
        For each message, adds an empty AI response message and links it in the interactions DataFrame.
        For each label, replaces it with the Questiontype instance with the same name.
        """

        print(f"Loading xlsx file: {file_path}")


        df = pd.read_excel(file_path)

        # Replace label string with Questiontype instance with the same name
        name_to_question_type = {type.name: type for type in question_types}
        df["question_type_by_Thom"] = df["question_type_by_Thom"].map(name_to_question_type)
        df["question_type_by_Stijn"] = df["question_type_by_Stijn"].map(name_to_question_type)

        n = len(df)
        # Create messages DataFrame: user messages
        messages = pd.DataFrame(
            {
                "message_id": range(n),
                "body": df["question"],
                "sender": "user",
                "automated": False,
            }
        )
        # Add empty AI responses
        ai_messages = pd.DataFrame(
            {
                "message_id": range(n, 2 * n),
                "body": "",
                "sender": "ai",
                "automated": False,
            }
        )
        messages = pd.concat([messages, ai_messages], ignore_index=True)

        # Create interactions DataFrame with answer_id
        interactions = pd.DataFrame(
            {
                "interaction_id": range(n),
                "question_id": range(n),
                "answer_id": range(n, 2 * n),
                "question_type_by_Thom": df["question_type_by_Thom"],
                "question_type_by_Stijn": df["question_type_by_Stijn"],
            }
        )

        data["messages"] = messages
        data["interactions"] = interactions
        print(
            f"Loaded {n} rows into DataFrames 'messages' and 'interactions' (with empty AI responses)"
        )

    return load_cld_questions
