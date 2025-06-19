from typing import Callable, Dict

import numpy as np
import pandas as pd

import chatbot
from enums import LearningGoal, QuestionPurpose, QuestionType


def add_interactions_df(data: Dict[str, pd.DataFrame]) -> None:
    """
    Create a DataFrame of interactions from the messages DataFrame.
    Each interaction is a pair of consecutive messages where the first is a question and the second is an answer.
    The resulting DataFrame has columns 'id', 'question_id' and 'answer_id' referring to the original DataFrame indices.
    """

    # Get all messages for each user_id and create interaction pairs within each user
    messages = data["messages"]
    interactions = []
    for user_id in messages["user_id"].unique():
        user_msgs = messages[messages["user_id"] == user_id].sort_values("datetime")
        for i in range(len(user_msgs) - 1):
            row_q = user_msgs.iloc[i]
            row_a = user_msgs.iloc[i + 1]
            is_question = not row_q["automated"]
            is_answer = row_a["automated"]
            if is_question and is_answer:
                interactions.append(
                    {
                        "user_id": user_id,
                        "question_id": row_q["message_id"],
                        "answer_id": row_a["message_id"],
                    }
                )

    interactions_df = pd.DataFrame(interactions)
    interactions_df.insert(0, "interaction_id", range(len(interactions_df)))
    data["interactions"] = interactions_df


def add_waiting_time_to_interactions(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add waiting time (delta between question and answer) to the interactions DataFrame.
    """

    interactions = data["interactions"]
    messages = data["messages"]

    # Merge interactions with messages to get question datetimes
    merged = interactions.merge(
        messages[["message_id", "datetime"]],
        left_on="question_id",
        right_on="message_id",
        how="left",
        suffixes=("", "_question"),
    ).rename(columns={"datetime": "question_datetime"})
    # Merge again to get answer datetimes
    merged = merged.merge(
        messages[["message_id", "datetime"]],
        left_on="answer_id",
        right_on="message_id",
        how="left",
        suffixes=("", "_answer"),
    ).rename(columns={"datetime": "answer_datetime"})

    # Calculate waiting time
    merged["waiting_time"] = (
        merged["answer_datetime"] - merged["question_datetime"]
    ).dt.total_seconds()

    # Assign back to interactions DataFrame
    interactions["waiting_time"] = merged["waiting_time"]
    data["interactions"] = interactions


def add_active_file(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add active file to the messages DataFrame.
    """

    messages = data["messages"]
    edits = data["edits"]
    edits = edits[
        edits["filename"].notnull()
    ]  # Skip edits that do not have a file associated

    active_files = []
    for id, msg in messages.iterrows():
        edits_after = edits[
            (edits["datetime"] > msg["datetime"]) & (edits["user_id"] == msg["user_id"])
        ]
        if not edits_after.empty:
            first_edit = edits_after.sort_values("datetime").iloc[0]
            active_files.append(first_edit["filename"])
        else:
            active_files.append(None)

    messages["active_file_after"] = active_files
    data["messages"] = messages


def add_interaction_type(
    question_types: list[QuestionType], not_detected_type: QuestionType
) -> Callable[[Dict[str, pd.DataFrame]], None]:
    def add_question_type(data: Dict[str, pd.DataFrame]) -> None:
        """
        Add question type to the interactions DataFrame by asking the chatbot for each question body.
        """

        messages = data["messages"]
        interactions = data["interactions"]

        # Merge to get question body for each interaction
        merged = interactions.merge(
            messages[["message_id", "body"]],
            left_on="question_id",
            right_on="message_id",
            how="left",
            suffixes=("", "_question"),
        )

        question_types_explanation = "\n".join(
            [
                f"{question_type.name}: {question_type.description}"
                for question_type in question_types
            ]
        )

        def prompt_fn(row):
            question_body = row["body"]
            return (
                "Let's work this out in a step by step way to be sure we have the right answer.\n"
                "What is the question type of the following question of a student?\n"
                "Format final line as: The question is of type: [TYPE]\n\n"
                "Here are the available types: \n"
                f"{question_types_explanation}\n\n"
                f"Now classify the following message:\n'''\n{question_body}\n'''\n"
            )

        def extract_fn(response, row):
            last_sentence = response.split("\n")[-1].lower().strip()
            detected_types = [
                qt for qt in question_types if qt.name.lower() in last_sentence
            ]
            if len(detected_types) == 1:
                return detected_types[0]
            raise ValueError("No valid question type detected")

        merged = chatbot.add_column_through_chatbot(
            merged,
            column_name="question_type_by_ai",
            generate_prompt_fn=prompt_fn,
            extract_data_fn=extract_fn,
            default_value=not_detected_type,
            max_retries=3,
        )

        interactions["question_type_prompt"] = merged["question_type_by_ai_prompt"]
        interactions["question_type_response"] = merged["question_type_by_ai_response"]
        interactions["question_type_by_ai"] = merged["question_type_by_ai"]

    return add_question_type


def add_interaction_purpose(
    question_purposes: list[QuestionPurpose],
) -> Callable[[Dict[str, pd.DataFrame]], None]:
    def add_question_purpose(data: Dict[str, pd.DataFrame]) -> None:
        """
        Add question purpose to the interactions DataFrame by asking the chatbot for each question body.
        """

        messages = data["messages"]
        interactions = data["interactions"]

        # Merge to get question and answer body for each interaction
        merged = interactions.merge(
            messages[["message_id", "body"]],
            left_on="question_id",
            right_on="message_id",
            how="left",
            suffixes=("", "_question"),
        ).rename(columns={"body": "question_body"})
        merged = merged.merge(
            messages[["message_id", "body"]],
            left_on="answer_id",
            right_on="message_id",
            how="left",
            suffixes=("", "_answer"),
        ).rename(columns={"body": "answer_body"})

        question_purposes_explanation = "\n - ".join(
            [
                f"{question_purpose.name}: {question_purpose.description}"
                for question_purpose in question_purposes
            ]
        )

        def prompt_fn(row):
            question_body = row["question_body"]
            answer_body = row["answer_body"]
            return (
                "Let's work this out in a step by step way to be sure we have the right answer.\n"
                "What is the question purpose of the following question of a student?\n"
                "Format final line as: The question is of purpose: [PURPOSE]\n\n"
                "These are the possible purposes:\n"
                f"{question_purposes_explanation}\n"
                f"Classify the following message:\n'''\n{question_body}\n'''\n"
                f"AI response:\n'''\n{answer_body}\n'''\n"
            )

        def extract_fn(response, row):
            last_sentence = response.lower().strip().split("\n")[-1]
            detected_purposes = [
                qp for qp in question_purposes if qp.name.lower() in last_sentence
            ]
            if len(detected_purposes) == 1:
                return detected_purposes[0]
            raise ValueError("No valid question purpose detected")

        merged = chatbot.add_column_through_chatbot(
            merged,
            column_name="question_purpose_by_ai",
            generate_prompt_fn=prompt_fn,
            extract_data_fn=extract_fn,
            default_value=None,
            max_retries=3,
        )

        interactions["question_purpose_by_ai_prompt"] = merged[
            "question_purpose_by_ai_prompt"
        ]
        interactions["question_purpose_by_ai_response"] = merged[
            "question_purpose_by_ai_response"
        ]
        interactions["question_purpose_by_ai"] = merged["question_purpose_by_ai"]

        def get_purpose_from_type(qtype: QuestionType):
            return qtype.question_purpose

        interactions["question_purpose_by_question_type"] = interactions[
            "question_type_by_ai"
        ].apply(get_purpose_from_type)

    return add_question_purpose


def add_interaction_learning_goals(
    learning_goals: list[LearningGoal],
) -> Callable[[Dict[str, pd.DataFrame]], None]:
    def add_question_learning_goals(data: Dict[str, pd.DataFrame]) -> None:
        """
        Add learning goals to the interactions DataFrame by asking the chatbot for each question body.
        """

        messages = data["messages"]
        interactions = data["interactions"]

        # Merge to get question and answer body for each interaction
        merged = interactions.merge(
            messages[["message_id", "body"]],
            left_on="question_id",
            right_on="message_id",
            how="left",
            suffixes=("", "_question"),
        ).rename(columns={"body": "question_body"})
        merged = merged.merge(
            messages[["message_id", "body"]],
            left_on="answer_id",
            right_on="message_id",
            how="left",
            suffixes=("", "_answer"),
        ).rename(columns={"body": "answer_body"})

        learning_goal_explanations = "\n".join(
            [f"{e.name}: {e.description}" for e in learning_goals]
        )

        def prompt_fn(row):
            question_body = row["question_body"]
            answer_body = row["answer_body"]
            return (
                "Let's work this out in a step by step way to be sure we have the right answer.\n"
                "About which learning goal is the following question of a student?\n"
                "Format final line as: The question has learning goals: [learning goals]\n\n"
                f"Available learning goals:\n{learning_goal_explanations}\n"
                f"Classify the following message:\n'''\n{question_body}\n'''\n"
                f"AI response:\n'''\n{answer_body}\n'''\n"
            )

        def extract_fn(response, row):
            last_sentence = response.lower().strip().split("\n")[-1]
            detected_goals = [
                lg for lg in learning_goals if lg.name.lower() in last_sentence
            ]
            if len(detected_goals) > 0:
                return detected_goals
            raise ValueError("No valid learning goal detected")

        merged = chatbot.add_column_through_chatbot(
            merged,
            column_name="question_learning_goals",
            generate_prompt_fn=prompt_fn,
            extract_data_fn=extract_fn,
            default_value=[],
            max_retries=3,
        )

        interactions["question_learning_goals"] = merged["question_learning_goals"]
        interactions["question_learning_goals_prompt"] = merged[
            "question_learning_goals_prompt"
        ]
        interactions["question_learning_goals_response"] = merged[
            "question_learning_goals_response"
        ]

    return add_question_learning_goals


def add_increase_in_success_rate(
    data: Dict[str, pd.DataFrame],
) -> None:
    """
    Add increase in success rate to the interactions DataFrame.
    """

    interactions = data["interactions"]
    messages = data["messages"]
    users = data["users"]

    # Merge the two DataFrames on the question index
    merged = interactions.merge(
        messages,
        left_on="question_id",
        right_on="message_id",
        how="left",
        suffixes=("", "_question"),
    )

    increases = []
    for id, interaction_row in merged.iterrows():
        # Get relevant data
        user_id = interaction_row["user_id"]
        question_goals = interaction_row["question_learning_goals"]
        datetime = interaction_row["datetime"]
        user_row = users[users["user_id"] == user_id]
        if len(user_row) != 1:
            raise ValueError(
                f"User {user_id} not found or multiple rows found in users DataFrame."
            )
        user_row = user_row.iloc[0]

        # Calculate ratios
        before_ratios = []
        after_ratios = []
        for goal in question_goals:
            column = f"{goal.name}_series"
            learning_goal_time_series = user_row[column]
            before = learning_goal_time_series[
                learning_goal_time_series["datetime"] < datetime
            ]
            after = learning_goal_time_series[
                learning_goal_time_series["datetime"] > datetime
            ]

            def ratio(df):
                if df.empty:
                    return None
                good = (df["result"] == True).sum()
                total = len(df)
                return good / total if total > 0 else None

            before_ratio = ratio(before)
            after_ratio = ratio(after)
            if before_ratio is not None and after_ratio is not None:
                before_ratios.append(before_ratio)
                after_ratios.append(after_ratio)

        if before_ratios and after_ratios:
            # Average if multiple goals
            increase = sum(after_ratios) / len(after_ratios) - sum(before_ratios) / len(
                before_ratios
            )
            increases.append(increase)
        else:
            increases.append(None)

    interactions["increase_in_success_rate"] = increases
    data["interactions"] = interactions


def add_time_until_next_interaction(
    data: Dict[str, pd.DataFrame],
) -> None:
    """
    Add time until next interaction to the interactions DataFrame.
    This calculates the time difference between the current interaction and the next interaction for each user.
    """

    interactions = data["interactions"]

    # Merge to get question datetimes for sorting
    merged = interactions.merge(
        data["messages"][["message_id", "datetime"]],
        left_on="question_id",
        right_on="message_id",
        how="left",
    )

    # Sort by user and datetime
    merged = merged.sort_values(["user_id", "datetime"]).reset_index(drop=True)

    # Calculate time until next interaction per user
    merged["time_until_next_interaction"] = (
        merged.groupby("user_id")["datetime"].shift(-1) - merged["datetime"]
    )
    # Convert timedelta to seconds, replace NaT with np.nan
    merged["time_until_next_interaction"] = merged["time_until_next_interaction"].apply(
        lambda x: x.total_seconds() if pd.notnull(x) else np.nan
    )

    # Assign back to interactions DataFrame
    interactions["time_until_next_interaction"] = merged["time_until_next_interaction"]
    data["interactions"] = interactions


def add_time_until_next_edit(
    data: Dict[str, pd.DataFrame],
) -> None:
    """
    Add time until next edit to the interactions DataFrame.
    This calculates the time difference between the current interaction and the next edit for each user.
    """
    interactions = data["interactions"]
    edits = data["edits"]

    # Merge to get question datetimes for sorting
    merged = interactions.merge(
        data["messages"][["message_id", "datetime"]],
        left_on="question_id",
        right_on="message_id",
        how="left",
    )

    # Sort edits by user and datetime
    edits_sorted = edits.sort_values(["user_id", "datetime"]).reset_index(drop=True)

    # For each interaction, find the next edit for the same user after the question datetime
    time_until_next_edit = []
    for idx, row in merged.iterrows():
        user_id = row["user_id"]
        question_time = row["datetime"]
        user_edits = edits_sorted[
            (edits_sorted["user_id"] == user_id)
            & (edits_sorted["datetime"] > question_time)
        ]
        if not user_edits.empty:
            next_edit_time = user_edits.iloc[0]["datetime"]
            delta = next_edit_time - question_time
            if pd.isnull(delta):
                time_until_next_edit.append(np.nan)
            else:
                time_until_next_edit.append(delta.total_seconds())
        else:
            time_until_next_edit.append(np.nan)

    # Fill any NaT values in the results with np.nan to avoid issues in downstream processing
    interactions["time_until_next_edit"] = pd.Series(time_until_next_edit)
    data["interactions"] = interactions


def add_time_until_next_execution(
    data: Dict[str, pd.DataFrame],
) -> None:
    """
    Add time until next execution to the interactions DataFrame.
    This calculates the time difference between the current interaction and the next execution for each user.
    """
    interactions = data["interactions"]
    executions = data["executions"]

    # Merge to get question datetimes for sorting
    merged = interactions.merge(
        data["messages"][["message_id", "datetime"]],
        left_on="question_id",
        right_on="message_id",
        how="left",
    )

    # Sort executions by user and datetime
    executions_sorted = executions.sort_values(["user_id", "datetime"]).reset_index(
        drop=True
    )

    # For each interaction, find the next execution for the same user after the question datetime
    time_until_next_execution = []
    for idx, row in merged.iterrows():
        user_id = row["user_id"]
        question_time = row["datetime"]
        user_executions = executions_sorted[
            (executions_sorted["user_id"] == user_id)
            & (executions_sorted["datetime"] > question_time)
        ]
        if not user_executions.empty:
            next_execution_time = user_executions.iloc[0]["datetime"]
            delta = next_execution_time - question_time
            if pd.isnull(delta):
                time_until_next_execution.append(np.nan)
            else:
                time_until_next_execution.append(delta.total_seconds())
        else:
            time_until_next_execution.append(np.nan)

    # Fill any NaT values in the results with np.nan to avoid issues in downstream processing
    interactions["time_until_next_execution"] = pd.Series(time_until_next_execution)
    data["interactions"] = interactions


def add_interaction_overview_df(data: Dict[str, pd.DataFrame]) -> None:
    """
    Merge all DataFrames that have an 'interaction_id' column into a single overview DataFrame.
    Start with the 'interactions' table (using 'id' as the key), then merge all others on 'interaction_id'.
    The resulting DataFrame is stored as 'interaction_overview' in the data dict.
    """

    overview_df = data["interactions"].copy()

    overview_df = overview_df.merge(
        data["messages"],
        left_on="question_id",
        right_on="message_id",
        how="left",
        suffixes=(None, f"_question"),
    )

    overview_df = overview_df.merge(
        data["messages"],
        left_on="answer_id",
        right_on="message_id",
        how="left",
        suffixes=(None, f"_answer"),
    )

    data["interaction_overview"] = overview_df
