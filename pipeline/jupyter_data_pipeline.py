import os
from typing import Callable, Dict, List

import pandas as pd

from correlations.correlation_plots import plot_violin_plot
from enums import get_learning_goals, get_question_purposes, get_question_types
from executions.execution_analyser import (
    add_execution_overview_df,
    add_execution_success,
    add_file_version_id,
    add_id_of_previous_executed_file_version,
    add_surrounding_executions,
)
from executions.execution_error_analyser import (
    add_error_learning_goal_by_ai_detection,
    add_error_learning_goal_by_error_pattern_detection,
)
from executions.execution_success_analyser import (
    add_constructs_of_added_code,
    add_execution_successes_df,
    add_learning_goals_of_added_code,
    add_line_numbers_of_new_code
)
from interactions.interaction_analyser import (
    add_active_file,
    add_increase_in_success_rate,
    add_interaction_learning_goals,
    add_interaction_overview_df,
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
from pipeline.pipeline import run_pipeline
from users.basic_user_analyser import (
    add_average_learning_goals_success,
    add_basic_statistics_to_users,
    add_learning_goals_result_series,
    add_users_dataframe,
)
from writer.excel import write_to_excel


def run_jupyter_data_pipeline():

    # Load all tables into a dictionary
    VOLUMES_DATA_LOCATION = os.getenv("VOLUMES_DATA_LOCATION")
    LOGS_DATA_LOCATION = os.getenv("LOGS_DATA_LOCATION")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR")
    if not VOLUMES_DATA_LOCATION or not LOGS_DATA_LOCATION or not OUTPUT_DIR:
        raise ValueError(
            "VOLUMES_DATA_LOCATION or LOGS_DATA_LOCATION or OUTPUT_DIR not set in .env"
        )
    FILTER_USERNAME = os.getenv("FILTER_USERNAME")

    # Get enums
    question_types = get_question_types()
    question_purposes = get_question_purposes()
    learning_goals = get_learning_goals()

    # Define your pipeline steps
    pipeline_steps: List[Callable[[Dict[str, pd.DataFrame]], None]] = [
        load_jupyter_log(LOGS_DATA_LOCATION.split(","), FILTER_USERNAME),
        # executions
        add_execution_success,
        add_file_version_id,
        add_id_of_previous_executed_file_version,
        add_surrounding_executions,
        # execution_success
        add_execution_successes_df,
        add_line_numbers_of_new_code,
        add_constructs_of_added_code,
        add_learning_goals_of_added_code(learning_goals),
        # execution_errors
        add_error_learning_goal_by_error_pattern_detection(learning_goals),
        add_error_learning_goal_by_ai_detection(learning_goals),
        # Edits
        # messages
        load_chat_log(VOLUMES_DATA_LOCATION.split(","), FILTER_USERNAME),
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
        add_basic_statistics_to_users,
        add_learning_goals_result_series(learning_goals),
        add_average_learning_goals_success(learning_goals),
        # Interactions part 2
        add_increase_in_success_rate,
        # overview
        add_execution_overview_df,
        add_interaction_overview_df,
        # Plots
        plot_violin_plot(
            "interactions",
            "question_learning_goals",
            "increase_in_success_rate",
            OUTPUT_DIR,
        ),
        plot_violin_plot(
            "interactions", "question_purpose", "increase_in_success_rate", OUTPUT_DIR
        ),
        plot_violin_plot(
            "interactions", "question_type", "increase_in_success_rate", OUTPUT_DIR
        ),
        # Save to Excel
        write_to_excel(f"{OUTPUT_DIR}/jupyter_data.xlsx"),
    ]

    # Run the pipeline
    run_pipeline(pipeline_steps)
