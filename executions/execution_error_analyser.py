from typing import Dict, Optional

import pandas as pd

import chatbot
from enums import LearningGoal


def add_learning_goal_in_error_pattern_detection(learning_goals: list[LearningGoal]):
    def add_learning_goal_in_error_pattern_detection(
        data: Dict[str, pd.DataFrame],
    ) -> None:
        """
        For each execution error, check for each learning goal if it is applied in the code that caused the error.
        """
        
        execution_errors_df = data["execution_errors"]
        executions_df = data["executions"]
        file_versions_df = data["file_versions"]

        # Merge execution_errors with executions on execution_id
        merged = execution_errors_df.merge(
            executions_df[["id", "file_version_id"]],
            left_on="execution_id",
            right_on="id",
            how="left",
        )
        # Merge with file_versions to get the code and file
        merged = merged.merge(
            file_versions_df.rename(columns={"id": "file_version_id"})[
                ["file_version_id", "file", "code"]
            ],
            on="file_version_id",
            how="left",
        )

        # Check for each learning goal if it is applied in the code that caused the error
        def detect_learning_goals(row):
            matched_goals = []
            for learning_goal in learning_goals:
                if learning_goal.found_in_error(
                    error_name=row["error_name"],
                    traceback=row["traceback"],
                    code=row["code"],
                ):
                    matched_goals.append(learning_goal)
            return matched_goals

        execution_errors_df["learning_goals_in_error_pattern_detection"] = merged.apply(
            detect_learning_goals, axis=1
        )

    return add_learning_goal_in_error_pattern_detection


def add_learning_goals_in_error_ai_detection(learning_goals: list[LearningGoal]):
    def add_learning_goals_in_error_ai_detection(data: Dict[str, pd.DataFrame]) -> None:
        """
        For each execution error, use the chatbot to classify the error into a learning goal using the code and error information.
        Adds a column 'learning_goal_in_error_ai' with the detected learning goal name (or None if not detected).
        """

        execution_errors_df = data["execution_errors"]
        executions_df = data["executions"]
        file_versions_df = data["file_versions"]

        # Merge execution_errors with executions on execution_id
        merged = execution_errors_df.merge(
            executions_df[["id", "file_version_id"]],
            left_on="execution_id",
            right_on="id",
            how="left",
        )
        # Merge with file_versions to get the code and file
        merged = merged.merge(
            file_versions_df.rename(columns={"id": "file_version_id"})[
                ["file_version_id", "file", "code"]
            ],
            on="file_version_id",
            how="left",
        )

        learning_goals_string = "\n".join(
            [f"- {goal.name}: {goal.description}" for goal in learning_goals]
        )

        def detect_learning_goal_ai(row: pd.Series) -> list[str]:
            code = row["code"]
            error_value = row["error_value"]
            traceback = row["traceback"]
            query = (
                "You are an instructor tasked with analyzing a programming mistake.\n"
                " Your goal is to determine which learning goals are relevant to the error message. You will be provided with the error message, the corresponding code, and a list of learning goals—each with a name and a brief explanation.\n"
                "Please reason step-by-step to arrive at your conclusion\n"
                "On the last line of your response, write your final classification in exactly this format:\n"
                "'The learning goals are [list of names of learning goals]'.\n\n"
                f"Learning goals:\n{learning_goals_string}\n\n"
                f"Student code:\n'''\n{code}\n'''\n"
                f"Error message:\n'''\n{error_value}\n{traceback}\n'''\n"
            )
            for i in range(3):
                if i == 0:
                    response = chatbot.ask_question(query).lower().strip()
                else:
                    response = chatbot.ask_question_without_cache(query).lower().strip()

                last_sentence = response.split("\n")[-1]
                detected_learning_goals = []
                for goal in learning_goals:
                    if goal.name.lower() in last_sentence:
                        detected_learning_goals.append(goal)

                if len(detected_learning_goals) > 0:
                    return detected_learning_goals

            return []

        # Use a list comprehension for type safety
        execution_errors_df["learning_goals_in_error_ai_detection"] = merged.apply(
            detect_learning_goal_ai, axis=1
        )

    return add_learning_goals_in_error_ai_detection
