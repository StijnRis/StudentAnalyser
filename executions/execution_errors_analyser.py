from typing import Dict

import pandas as pd

import chatbot
from enums import LearningGoal
from executions.execution_error_cols import ExecutionErrorCols
from executions.execution_cols import ExecutionsCols
from file_versions.file_version_cols import FileVersionsCols


def add_error_learning_goal_by_error_pattern_detection(
    learning_goals: list[LearningGoal],
):
    def add_error_learning_goal_by_error_pattern_detection(
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
            executions_df[
                [ExecutionsCols.ID.value, ExecutionsCols.FILE_VERSION_ID.value]
            ],
            left_on=ExecutionErrorCols.EXECUTION_ID.value,
            right_on=ExecutionsCols.ID.value,
            how="left",
        )
        # Merge with file_versions to get the code and file
        merged = merged.merge(
            file_versions_df.rename(columns={"id": "file_version_id"})[
                [
                    "file_version_id",
                    FileVersionsCols.FILE.value,
                    FileVersionsCols.CODE.value,
                ]
            ],
            left_on=ExecutionErrorCols.ID.value,
            right_on="file_version_id",
            how="left",
        )

        # Check for each learning goal if it is applied in the code that caused the error
        def detect_learning_goals(row):
            matched_goals = []
            for learning_goal in learning_goals:
                if learning_goal.found_in_error(
                    error_name=row[ExecutionErrorCols.ERROR_NAME.value],
                    traceback=row[ExecutionErrorCols.TRACEBACK.value],
                    code=row[FileVersionsCols.CODE.value],
                ):
                    matched_goals.append(learning_goal)
            return matched_goals

        execution_errors_df[
            ExecutionErrorCols.LEARNING_GOALS_IN_ERROR_BY_ERROR_PATTERN_DETECTION.value
        ] = merged.apply(detect_learning_goals, axis=1)

    return add_error_learning_goal_by_error_pattern_detection


def add_error_learning_goal_by_ai_detection(learning_goals: list[LearningGoal]):
    def add_error_learning_goal_by_ai_detection(data: Dict[str, pd.DataFrame]) -> None:
        """
        For each execution error, use the chatbot to classify the error into a learning goal using the code and error information.
        Adds a column 'learning_goals_in_error_by_ai_detection' with the detected learning goal(s) (or None if not detected).
        """
        execution_errors_df = data["execution_errors"]
        executions_df = data["executions"]
        file_versions_df = data["file_versions"]

        merged = execution_errors_df.merge(
            executions_df[["id", "file_version_id"]],
            left_on="execution_id",
            right_on="id",
            how="left",
        )
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

        def prompt_fn(row):
            code = row[FileVersionsCols.CODE.value]
            error_value = row[ExecutionErrorCols.ERROR_VALUE.value]
            traceback = row[ExecutionErrorCols.TRACEBACK.value]
            return (
                "You are an instructor tasked with analyzing a programming mistake.\n"
                "Your goal is to determine which learning goals are relevant to the error message. You will be provided with the error message, the corresponding code, and a list of learning goals—each with a name and a brief explanation.\n"
                "Please reason step-by-step to arrive at your conclusion\n"
                "On the last line of your response, write your final classification in exactly this format:\n"
                "'The learning goals are [list of names of learning goals]'.\n\n"
                f"Learning goals:\n{learning_goals_string}\n\n"
                f"Student code:\n'''\n{code}\n'''\n"
                f"Error message:\n'''\n{error_value}\n{traceback}\n'''\n"
            )

        def extract_fn(response, row):
            last_sentence = response.split("\n")[-1]
            detected_learning_goals = [
                goal
                for goal in learning_goals
                if goal.name.lower() in last_sentence.lower()
            ]
            if len(detected_learning_goals) > 0:
                return detected_learning_goals
            raise ValueError("No valid learning goal detected")

        merged = chatbot.add_column_through_chatbot(
            merged,
            column_name=ExecutionErrorCols.LEARNING_GOALS_IN_ERROR_BY_AI_DETECTION.value,
            prompt_fn=prompt_fn,
            extract_fn=extract_fn,
            max_retries=3,
        )
        execution_errors_df[
            ExecutionErrorCols.LEARNING_GOALS_IN_ERROR_BY_AI_DETECTION.value
        ] = merged[ExecutionErrorCols.LEARNING_GOALS_IN_ERROR_BY_AI_DETECTION.value]
        execution_errors_df[
            ExecutionErrorCols.LEARNING_GOALS_IN_ERROR_BY_AI_DETECTION_PROMPT.value
        ] = merged[
            ExecutionErrorCols.LEARNING_GOALS_IN_ERROR_BY_AI_DETECTION_PROMPT.value
        ]
        execution_errors_df[
            ExecutionErrorCols.LEARNING_GOALS_IN_ERROR_BY_AI_DETECTION_RESPONSE.value
        ] = merged[
            ExecutionErrorCols.LEARNING_GOALS_IN_ERROR_BY_AI_DETECTION_RESPONSE.value
        ]

    return add_error_learning_goal_by_ai_detection


def add_error_learning_goal_by_user_fix_detection(
    learning_goals: list[LearningGoal],
):
    def add_error_learning_goal_by_user_fix_detection(
        data: Dict[str, pd.DataFrame],
    ) -> None:
        """
        For each execution error, check for each learning goal if it is applied in the code that caused the error.
        """

        execution_errors_df = data["execution_errors"]
        executions_df = data["executions"]
        file_versions_df = data["file_versions"]

        # Make table
        merged = execution_errors_df.merge(
            executions_df[["id", "file_version_id"]],
            left_on="execution_id",
            right_on="id",
            how="left",
        )
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
                    error_name=row[ExecutionErrorCols.ERROR_NAME.value],
                    traceback=row[ExecutionErrorCols.TRACEBACK.value],
                    code=row[FileVersionsCols.CODE.value],
                ):
                    matched_goals.append(learning_goal)
            return matched_goals

        execution_errors_df[
            ExecutionErrorCols.LEARNING_GOALS_IN_ERROR_BY_ERROR_PATTERN_DETECTION.value
        ] = merged.apply(detect_learning_goals, axis=1)

    return add_error_learning_goal_by_error_pattern_detection
