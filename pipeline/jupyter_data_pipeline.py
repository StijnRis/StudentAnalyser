import os
from typing import Callable, Dict, List

import pandas as pd

from correlations.correlation_plots import plot_violin_plot
from enums import get_learning_goals, get_question_purposes, get_question_types
from executions.execution_analyser import (
    add_execution_overview_df,
    add_execution_success,
    add_file_version_id,
    add_surrounding_executions,
)
from executions.execution_error_analyser import (
    add_error_learning_goal_by_error_pattern_detection,
    add_user_fix_analysis,
)
from executions.execution_success_analyser import (
    add_execution_successes_df,
    add_new_code_analysis,
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
from loader.loader_pipeline import generate_loader_pipeline
from messages.message_analyser import (
    add_code_in_message,
    add_included_code_snippets,
    add_message_language,
    add_message_length,
)
from pipeline.pipeline import run_pipeline
from users.user_analyser import (
    add_basic_learning_goals_statistics,
    add_basic_statistics_to_users,
    add_bayesian_knowledge_tracing,
    add_learning_goals_result_series,
    add_moving_average,
)
from writer.excel import write_to_excel


def run_jupyter_data_pipeline():

    # Load all tables into a dictionary
    METADATA_FOR_ANALYZER_PATH = os.getenv("METADATA_FOR_ANALYZER_PATH")
    BASE_DATA_PATH = os.getenv("BASE_DATA_PATH")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR")
    FILTER_USERNAME = os.getenv("FILTER_USERNAME", None)
    if not METADATA_FOR_ANALYZER_PATH or not OUTPUT_DIR or not BASE_DATA_PATH:
        raise ValueError(".env not set up correctly.")

    # Get enums
    question_types = get_question_types()
    question_purposes = get_question_purposes()
    learning_goals = get_learning_goals()

    # Define your pipeline steps
    pipeline_steps: List[Callable[[Dict[str, pd.DataFrame]], None]] = [
        *generate_loader_pipeline(
            BASE_DATA_PATH, METADATA_FOR_ANALYZER_PATH, FILTER_USERNAME
        ),
        # executions
        add_execution_success,
        add_file_version_id,
        add_surrounding_executions,
        # execution_success
        add_execution_successes_df,
        add_new_code_analysis(learning_goals),
        # execution_errors
        add_error_learning_goal_by_error_pattern_detection(learning_goals),
        # add_error_learning_goal_by_ai_detection(learning_goals),
        add_user_fix_analysis(learning_goals),
        # edits
        # messages
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
        add_basic_statistics_to_users,
        add_learning_goals_result_series(learning_goals),
        add_basic_learning_goals_statistics(learning_goals),
        add_bayesian_knowledge_tracing(learning_goals),
        add_moving_average(learning_goals, window_size=20),
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
