import os
from typing import Callable, Dict, List

import pandas as pd
from dotenv import load_dotenv

from correlations.correlation_plots import plot_violin_plot
from enums import get_learning_goals, get_question_purposes, get_question_types
from executions.execution_analyser import (
    add_execution_success,
    add_file_version_id,
    add_id_of_previous_executed_file_version,
)
from executions.execution_error_analyser import (
    add_learning_goal_in_error_pattern_detection,
    add_learning_goals_in_error_ai_detection,
)
from executions.execution_success_analyser import (
    add_constructs_of_added_code,
    add_execution_successes_df,
    add_learning_goals_of_added_code,
    add_line_numbers_of_new_code,
    add_prev_successful_executed_file_version_id,
)
from interactions.interaction_analyser import (
    add_active_file,
    add_increase_in_success_rate,
    add_interaction_learning_goals,
    add_interaction_purpose,
    add_interaction_type,
    add_interactions_df,
    add_waiting_time_to_interactions,
)
from loader.chatbot_log import load_chat_log
from loader.jupyter_log import load_jupyter_log
from messages.message_analyser import (
    add_code_in_message,
    add_included_code_snippets,
    add_message_language,
    add_message_length,
)
from users.basic_user_analyser import (
    add_average_learning_goals_success,
    add_learning_goals_result_series,
    add_users_dataframe,
)
from writer.excel import write_to_excel


def add_execution_overview_df(data: Dict[str, pd.DataFrame]) -> None:
    """
    Merge all DataFrames that have an 'execution_id' column into a single overview DataFrame.
    Start with the 'executions' table (using 'id' as the key), then merge all others on 'execution_id',
    and finally merge the file_versions table on 'file_version_id' from executions.
    The resulting DataFrame is stored as 'execution_overview' in the data dict.
    """

    overview_df = data["executions"].copy()

    # Merge file_versions on file_version_id from executions
    file_versions_df = data["file_versions"]
    overview_df = overview_df.merge(
        file_versions_df,
        left_on="file_version_id",
        right_on="id",
        how="left",
        suffixes=(None, "_file_version"),
    )

    # Merge all DataFrames with 'execution_id' except executions itself
    for key, df in data.items():
        if key == "executions":
            continue
        if "execution_id" in df.columns:
            overview_df = overview_df.merge(
                df,
                left_on="id",
                right_on="execution_id",
                how="left",
                suffixes=(None, f"_{key}"),
            )

    data["execution_overview"] = overview_df


def run_pipeline(data: Dict[str, pd.DataFrame], steps: List[Callable]) -> None:
    for step in steps:
        print(step.__name__)
        step(data)


def main():
    load_dotenv()
    # Load all tables into a dictionary
    volumes_data_location = os.getenv("VOLUMES_DATA_LOCATION")
    logs_data_location = os.getenv("LOGS_DATA_LOCATION")
    if not volumes_data_location or not logs_data_location:
        raise ValueError("VOLUMES_DATA_LOCATION or LOGS_DATA_LOCATION not set in .env")

    filter_username = os.getenv("FILTER_USERNAME")

    question_types = get_question_types()
    question_purposes = get_question_purposes()
    learning_goals = get_learning_goals()

    # Define your pipeline steps
    pipeline_steps: List[Callable[[Dict[str, pd.DataFrame]], None]] = [
        load_jupyter_log(logs_data_location.split(","), filter_username),
        # executions
        add_execution_success,
        add_file_version_id,
        add_id_of_previous_executed_file_version,
        # execution_success
        add_execution_successes_df,
        add_prev_successful_executed_file_version_id,
        add_line_numbers_of_new_code,
        add_constructs_of_added_code,
        add_learning_goals_of_added_code(learning_goals),
        # execution_errors
        add_learning_goal_in_error_pattern_detection(learning_goals),
        add_learning_goals_in_error_ai_detection(learning_goals),
        # Edits
        # messages
        load_chat_log(volumes_data_location.split(","), filter_username),
        add_code_in_message,
        add_message_length,
        add_included_code_snippets,
        add_message_language,
        add_active_file,
        # interactions
        add_interactions_df,
        add_waiting_time_to_interactions,
        add_interaction_type(question_types),
        add_interaction_purpose(question_purposes),
        add_interaction_learning_goals(learning_goals),
        # Users
        add_users_dataframe,
        add_learning_goals_result_series(learning_goals),
        add_average_learning_goals_success(learning_goals),
        # Interactions part 2
        add_increase_in_success_rate,
        # overview
        add_execution_overview_df,
        # Plots
        plot_violin_plot("interactions", "question_learning_goals", "increase_in_success_rate"),
        plot_violin_plot("interactions", "question_purpose", "increase_in_success_rate"),
        plot_violin_plot("interactions", "question_type", "increase_in_success_rate"),
    ]

    # Run the pipeline
    data: dict[str, pd.DataFrame] = {}
    run_pipeline(data, pipeline_steps)

    # Save the results to Excel
    write_to_excel(data, "output/result.xlsx")


if __name__ == "__main__":
    main()
