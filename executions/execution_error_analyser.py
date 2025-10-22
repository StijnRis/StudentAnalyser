from typing import Dict

import pandas as pd

import chatbot
from enums import LearningGoal
from executions.execution_utils import (
    convert_ast_nodes_to_strings,
    detect_learning_goals,
    get_ast_nodes_for_ranges,
    get_ranges_of_changed_code,
)


def add_cleaned_traceback(data: Dict[str, pd.DataFrame]) -> None:
    """
    Clean the traceback information in the execution errors DataFrame.
    Removes line numbers, file references, caret lines, and separator lines. Standardizes format.
    """
    import re

    execution_errors_df = data["execution_errors"]

    def traceback_no_formatting(traceback: str) -> str:
        return re.sub(
            r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]",
            "",
            traceback,
        )
    
    execution_errors_df["traceback_no_formatting"] = execution_errors_df[
        "traceback"
    ].apply(traceback_no_formatting)


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

        # Merge executions to get the file_version_id
        merged = execution_errors_df.merge(
            executions_df[["execution_id", "file_version_id"]],
            left_on="execution_id",
            right_on="execution_id",
            how="left",
        )
        # Merge file_versions to get the code and file
        merged = merged.merge(
            file_versions_df[
                [
                    "file_version_id",
                    "filename",
                    "code",
                ]
            ],
            left_on="file_version_id",
            right_on="file_version_id",
            how="left",
        )

        # Check for each learning goal if it is applied in the code that caused the error
        def detect_learning_goals(row):
            matched_goals = []
            for learning_goal in learning_goals:
                if learning_goal.found_in_error(
                    error_name=row["error_name"],
                    traceback=row["traceback_no_formatting"],
                    code=row["code"],
                    code_line=row["code_part"],
                ):
                    matched_goals.append(learning_goal)
            return matched_goals

        # Add code_line column
        def extract_code_part(row):
            import re

            yellow_code_pattern = re.compile(r"\x1b\[[0-9;]*43m(.*?)\x1b\[[0-9;]*49m")
            matches = yellow_code_pattern.findall(row["traceback"])
            return "".join(matches) if matches else ""

        merged["code_part"] = merged.apply(extract_code_part, axis=1)
        execution_errors_df["code_part"] = merged["code_part"]
        execution_errors_df["learning_goals_in_error_by_error_pattern_detection"] = (
            merged.apply(detect_learning_goals, axis=1)
        )

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
            executions_df[["execution_id", "file_version_id"]],
            left_on="execution_id",
            right_on="execution_id",
            how="left",
        )
        merged = merged.merge(
            file_versions_df[["file_version_id", "filename", "code"]],
            on="file_version_id",
            how="left",
        )
        learning_goals_string = "\n".join(
            [f"- {goal.name}: {goal.description}" for goal in learning_goals]
        )

        def prompt_fn(row):
            code = row["code"]
            error_value = row["error_value"]
            traceback = row["traceback_no_formatting"]
            return (
                "Let's work this out in a step by step way to be sure we have the right answer.\n"
                "What learning goals failed for the following error?\n"
                "Format final line as: The learning goals are : [list of name of learning goals]\n\n"
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
            column_name="learning_goals_in_error_by_ai",
            generate_prompt_fn=prompt_fn,
            extract_data_fn=extract_fn,
            default_value=[],
            max_retries=3,
        )
        execution_errors_df["learning_goals_in_error_by_ai"] = merged[
            "learning_goals_in_error_by_ai"
        ]
        execution_errors_df["learning_goals_in_error_by_ai_prompt"] = merged[
            "learning_goals_in_error_by_ai_prompt"
        ]
        execution_errors_df["learning_goals_in_error_by_ai_response"] = merged[
            "learning_goals_in_error_by_ai_response"
        ]

    return add_error_learning_goal_by_ai_detection


def add_error_learning_goal_by_user_fix(learning_goals: list[LearningGoal]):
    def add_error_learning_goal_by_user_fix(data: Dict[str, pd.DataFrame]) -> None:
        """
        For each execution error, compute:
            - line numbers of code changed in the next successful execution
            - AST constructs of the changed code
            - Learning goals applied in the changed code
        """
        file_versions_df = data["file_versions"]
        execution_errors_df = data["execution_errors"]
        executions_df = data["executions"]

        # Merge execution_errors with executions
        merged = execution_errors_df.merge(
            executions_df[
                [
                    "execution_id",
                    "file_version_id",
                    "next_success_file_version_id",
                ]
            ],
            left_on="execution_id",
            right_on="execution_id",
            how="left",
        )
        # Merge with file_versions to get code for current file version
        merged = merged.merge(
            file_versions_df[["file_version_id", "code"]],
            left_on="file_version_id",
            right_on="file_version_id",
            how="left",
            suffixes=("", "_file_version"),
        )
        # Merge with file_versions again to get next code
        merged = merged.merge(
            file_versions_df[["file_version_id", "code"]],
            left_on="next_success_file_version_id",
            right_on="file_version_id",
            how="left",
            suffixes=("", "_next_version"),
        )

        merged["code_next_version"] = merged["code_next_version"].fillna("")

        def compute_ranges_of_changed_code(row):
            code_next_version = row["code_next_version"]
            code_current = row["code"]
            return get_ranges_of_changed_code(code_current, code_next_version)

        def compute_changed_constructs(row):
            code_next_version = row["code_next_version"] or row["code"]
            ranges = row["ranges_of_changed_code_next_success"]
            return get_ast_nodes_for_ranges(code_next_version, ranges)
        
        def compute_constructs_as_strings(row):
            nodes = row["changed_constructs_next_success"]
            return convert_ast_nodes_to_strings(nodes)

        def compute_learning_goals_of_changed_code(row):
            constructs = row["changed_constructs_next_success"]
            return list(set(detect_learning_goals(constructs, learning_goals)))

        # Compute all columns in sequence with new names
        merged["ranges_of_changed_code_next_success"] = merged.apply(
            compute_ranges_of_changed_code, axis=1
        )
        execution_errors_df["ranges_of_changed_code_next_success"] = merged[
            "ranges_of_changed_code_next_success"
        ]

        merged["changed_constructs_next_success"] = merged.apply(
            compute_changed_constructs, axis=1
        ).astype(object)
        execution_errors_df["changed_constructs_next_success"] = merged[
            "changed_constructs_next_success"
        ]

        merged["changed_constructs_as_string_next_success"] = merged.apply(
            compute_constructs_as_strings, axis=1
        )
        execution_errors_df["changed_constructs_as_string_next_success"] = merged[
            "changed_constructs_as_string_next_success"
        ]

        merged["learning_goals_in_error_by_user_fix"] = merged.apply(
            compute_learning_goals_of_changed_code, axis=1
        )
        execution_errors_df["learning_goals_in_error_by_user_fix"] = merged[
            "learning_goals_in_error_by_user_fix"
        ]

        data["execution_errors"] = execution_errors_df

    return add_error_learning_goal_by_user_fix
