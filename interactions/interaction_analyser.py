from typing import Callable, Dict

import pandas as pd

import chatbot
from enums import LearningGoal, QuestionPurpose, QuestionType


def add_interactions_df(data: Dict[str, pd.DataFrame]) -> None:
    """
    Create a DataFrame of interactions from the messages DataFrame.
    Each interaction is a pair of consecutive messages where the first is a question and the second is an answer.
    The resulting DataFrame has columns 'id', 'question_id' and 'answer_id' referring to the original DataFrame indices.
    """

    # Get all messages for each username and create interaction pairs within each user
    messages = data["messages"]
    interactions = []
    for username in messages["username"].unique():
        user_msgs = messages[messages["username"] == username].sort_values("datetime")
        for i in range(len(user_msgs) - 1):
            row_q = user_msgs.iloc[i]
            row_a = user_msgs.iloc[i + 1]
            is_question = not row_q["automated"]
            is_answer = row_a["automated"]
            if is_question and is_answer:
                interactions.append(
                    {
                        "username": username,
                        "question_id": row_q["id"],
                        "answer_id": row_a["id"],
                    }
                )

    interactions_df = pd.DataFrame(interactions)
    interactions_df.insert(0, "id", range(len(interactions_df)))
    data["interactions"] = interactions_df


def add_waiting_time_to_interactions(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add waiting time (delta between question and answer) to the interactions DataFrame.
    """

    interactions = data["interactions"]
    messages = data["messages"]

    # Merge interactions with messages to get question datetimes
    merged = interactions.merge(
        messages[["id", "datetime"]],
        left_on="question_id",
        right_on="id",
        how="left",
        suffixes=("", "_question"),
    ).rename(columns={"datetime": "question_datetime"})
    # Merge again to get answer datetimes
    merged = merged.merge(
        messages[["id", "datetime"]],
        left_on="answer_id",
        right_on="id",
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
        edits["file"].notnull()
    ]  # Skip edits that do not have a file associated

    active_files = []
    for id, msg in messages.iterrows():
        edits_after = edits[
            (edits["datetime"] > msg["datetime"])
            & (edits["username"] == msg["username"])
        ]
        if not edits_after.empty:
            first_edit = edits_after.sort_values("datetime").iloc[0]
            active_files.append(first_edit["file"])
        else:
            active_files.append(None)

    messages["active_file_after"] = active_files
    data["messages"] = messages


def add_interaction_type(
    question_types: list[QuestionType],
) -> Callable[[Dict[str, pd.DataFrame]], None]:
    def add_question_type(data: Dict[str, pd.DataFrame]) -> None:
        """
        Add question type to the interactions DataFrame by asking the chatbot for each question body.
        """

        messages = data["messages"]
        interactions = data["interactions"]

        # Merge to get question body for each interaction
        merged = interactions.merge(
            messages[["id", "body"]],
            left_on="question_id",
            right_on="id",
            how="left",
            suffixes=("", "_question"),
        )

        question_types_list = []
        for _, row in merged.iterrows():
            question_body = row["body"]
            detected_types = []
            question_types_explanation = "\n".join(
                [
                    f"{question_type.name}: {question_type.description}"
                    for question_type in question_types
                ]
            )
            prompt = (
                "You are an expert question type classifier.\n"
                "For each incoming user message, follow this procedure:\n"
                "1. Carefully analyze the message and think through its intent.\n"
                "2. Compare it against the list of predefined question types below.\n"
                "3. On the final line, deliver the verdict in this format:\n"
                "   The question is of type: [TYPE]\n\n"
                "Here are the available types: \n"
                f"{question_types_explanation}\n\n"
                f"Now classify the following message:\n'''\n{question_body}\n'''\n"
            )
            for i in range(3):
                if i == 0:
                    response = chatbot.ask_question(prompt)
                else:
                    response = chatbot.ask_question_without_cache(prompt)
                response = response.lower().strip()
                last_sentence = response.split("\n")[-1]
                for question_type in question_types:
                    if question_type.name.lower() in last_sentence:
                        detected_types.append(question_type)
                if len(detected_types) == 1:
                    question_types_list.append(detected_types[0])
                    break
            else:
                question_types_list.append(None)

        interactions["question_type"] = question_types_list

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
            messages[["id", "body"]],
            left_on="question_id",
            right_on="id",
            how="left",
            suffixes=("", "_question"),
        ).rename(columns={"body": "question_body"})
        merged = merged.merge(
            messages[["id", "body"]],
            left_on="answer_id",
            right_on="id",
            how="left",
            suffixes=("", "_answer"),
        ).rename(columns={"body": "answer_body"})

        question_purposes_list = []
        for _, row in merged.iterrows():
            question_body = row["question_body"]
            answer_body = row["answer_body"]
            detected_purposes = []
            question_purposes_explanation = "\n - ".join(
                [
                    f"{question_purpose.name}: {question_purpose.description}"
                    for question_purpose in question_purposes
                ]
            )
            prompt = (
                "You are an expert question purpose classifier.\n"
                "For each incoming user message, follow this procedure:\n"
                "1. Carefully analyze the message and think through its intent.\n"
                "2. Compare it against the list of predefined question purposes below.\n"
                "3. On the final line, deliver the verdict in this format:\n"
                "   The question is of purpose: [PURPOSE]\n\n"
                "Here are the available purposes:\n"
                f"{question_purposes_explanation}\n"
                f"Classify the following message:\n'''\n{question_body}\n'''\n"
                f"AI response:\n'''\n{answer_body}\n'''\n"
            )
            for i in range(3):
                if i == 0:
                    response = chatbot.ask_question(prompt)
                else:
                    response = chatbot.ask_question_without_cache(prompt)
                response = response.lower().strip()
                last_sentence = response.split("\n")[-1]
                for question_purpose in question_purposes:
                    if question_purpose.name.lower() in last_sentence:
                        detected_purposes.append(question_purpose)
                if len(detected_purposes) == 1:
                    question_purposes_list.append(detected_purposes[0])
                    break
            else:
                question_purposes_list.append(None)
        interactions["question_purpose"] = question_purposes_list

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
            messages[["id", "body"]],
            left_on="question_id",
            right_on="id",
            how="left",
            suffixes=("", "_question"),
        ).rename(columns={"body": "question_body"})
        merged = merged.merge(
            messages[["id", "body"]],
            left_on="answer_id",
            right_on="id",
            how="left",
            suffixes=("", "_answer"),
        ).rename(columns={"body": "answer_body"})

        question_learning_goals_list = []
        for _, row in merged.iterrows():
            question_body = row["question_body"]
            answer_body = row["answer_body"]
            detected_goals = []
            learning_goal_explanations = "\n".join(
                [f"{e.name}: {e.description}" for e in learning_goals]
            )
            prompt = (
                "You are an expert question learning goal classifier.\n"
                "For each incoming user message, follow this procedure:\n"
                "1. Carefully analyze the message and think through its intent.\n"
                "2. Compare it against the list of predefined learning goals below.\n"
                "3. On the final line, deliver the verdict in this format:\n"
                "   The question is of purpose: [PURPOSE]\n\n"
                f"Available learning goals:\n{learning_goal_explanations}\n"
                f"Classify the following message:\n'''\n{question_body}\n'''\n"
                f"AI response:\n'''\n{answer_body}\n'''\n"
            )

            for i in range(3):
                if i == 0:
                    response = chatbot.ask_question(prompt)
                else:
                    response = chatbot.ask_question_without_cache(prompt)
                response = response.lower().strip()
                last_sentence = response.split("\n")[-1]
                for learning_goal in learning_goals:
                    if learning_goal.name.lower() in last_sentence:
                        detected_goals.append(learning_goal)
                if len(detected_goals) > 0:
                    question_learning_goals_list.append(detected_goals)
                    break
            else:
                question_learning_goals_list.append("")
        interactions["question_learning_goals"] = question_learning_goals_list

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
        right_on="id",
        how="left",
        suffixes=("", "_question"),
    )

    increases = []
    for id, interaction_row in merged.iterrows():
        # Get relevant data
        username = interaction_row["username"]
        question_goals = interaction_row["question_learning_goals"]
        datetime = interaction_row["datetime"]
        user_row = users[users["username"] == username]
        if len(user_row) != 1:
            raise ValueError(
                f"User {username} not found or multiple rows found in users DataFrame."
            )
        user_row = user_row.iloc[0]

        # Calculate ratios
        before_ratios = []
        after_ratios = []
        for goal in question_goals:
            column = f"learning_goal_result_series_{goal.name}"
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
