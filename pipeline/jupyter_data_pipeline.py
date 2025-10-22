import os
from typing import Callable, Dict, List

import pandas as pd

from enums import get_learning_goals, get_question_purposes, get_question_types
from executions.execution_analyser import (
    add_execution_overview_df,
    add_execution_success,
    add_file_version_id,
    add_surrounding_executions,
)
from executions.execution_error_analyser import (
    add_cleaned_traceback,
    add_error_learning_goal_by_error_pattern_detection,
    add_error_learning_goal_by_user_fix,
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
    add_time_until_next_edit,
    add_time_until_next_execution,
    add_time_until_next_interaction,
)
from loader.load_excel_file import (
    generate_load_labelled_questions,
    generate_load_labelled_traceback_errors,
)
from loader.loader_pipeline import generate_start_loader_pipeline
from messages.message_analyser import (
    add_code_in_message,
    add_included_code_snippets,
    add_message_language,
    add_message_length,
)
from pipeline.pipeline import run_pipeline
from plots.confusion_matrix import plot_confusion_matrix
from plots.correlation_matrix import plot_correlation_matrix
from plots.scatter_plot import plot_scatter_plot
from plots.violin_plot import plot_violin_plot
from timeline.timeline_analyser import add_timeline_df
from users.user_analyser import (
    add_aggregate_construct_series,
    add_aggregate_learning_goal_series,
    add_basic_interaction_statistics,
    add_basic_statistics_for_series,
    add_basic_user_statistics,
    add_bayesian_knowledge_tracing,
    add_construct_result_series,
    add_learning_goals_result_series,
    add_moving_average,
)
from writer.excel import write_to_excel


def run_jupyter_data_pipeline():
    # Load all tables into a dictionary
    METADATA_FOR_ANALYZER_PATH = os.getenv("METADATA_FOR_ANALYZER_PATH")
    BASE_DATA_PATH = os.getenv("BASE_DATA_PATH")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR")
    GROUPS = os.getenv("GROUPS")
    if not METADATA_FOR_ANALYZER_PATH:
        raise ValueError(".env misses METADATA_FOR_ANALYZER_PATH")
    if not BASE_DATA_PATH:
        raise ValueError(".env misses BASE_DATA_PATH")
    if not OUTPUT_DIR:
        raise ValueError(".env misses OUTPUT_DIR")
    if not GROUPS:
        raise ValueError(".env misses GROUPS")
    GROUPS = GROUPS.split(",")
    print(f"Running pipeline for groups: {GROUPS}")

    FILTER_USERNAMES = os.getenv("FILTER_USERNAMES", None)
    if FILTER_USERNAMES:
        FILTER_USERNAMES = FILTER_USERNAMES.split(",")
        print(f"Filtering usernames: {FILTER_USERNAMES}")
    else:
        FILTER_USERNAMES = None
        print("No username filtering applied.")

    # Get enums
    question_types = get_question_types()
    unknown_question_type = question_types[-1]
    if unknown_question_type.name != "Not detected":
        raise ValueError(
            "The last question type should be 'Not detected', but it is not. "
            "Please check the get_question_types function."
        )
    question_purposes = get_question_purposes()
    learning_goals = get_learning_goals()

    for group in GROUPS:
        print(f"Running pipeline for group: {group}")

        group_output_dir = os.path.join(OUTPUT_DIR, group)
        if FILTER_USERNAMES:
            group_output_dir = os.path.join(group_output_dir, "filtered")
        os.makedirs(group_output_dir, exist_ok=True)

        # Define your pipeline steps for this group
        pipeline_steps: List[Callable[[Dict[str, pd.DataFrame]], None]] = [
            # Load data
            *generate_start_loader_pipeline(
                BASE_DATA_PATH,
                METADATA_FOR_ANALYZER_PATH,
                FILTER_USERNAMES,
                group,
            ),
            # executions
            add_execution_success,
            add_file_version_id,
            add_surrounding_executions,
            # execution_success
            add_execution_successes_df,
            add_new_code_analysis(learning_goals),
            # execution_errors
            add_cleaned_traceback,
            *generate_load_labelled_traceback_errors(
                BASE_DATA_PATH, METADATA_FOR_ANALYZER_PATH
            ),
            add_error_learning_goal_by_error_pattern_detection(learning_goals),
            # add_error_learning_goal_by_ai_detection(learning_goals),
            add_error_learning_goal_by_user_fix(learning_goals),
            # edits
            # messages
            add_code_in_message,
            add_message_length,
            add_included_code_snippets,
            add_message_language,
            add_active_file,
            # interactions
            add_interactions_df,
            add_time_until_next_interaction,
            add_time_until_next_edit,
            add_time_until_next_execution,
            # add_waiting_time_to_interactions,
            add_interaction_type(question_types, unknown_question_type),
            add_interaction_purpose(question_purposes),
            add_interaction_learning_goals(learning_goals),
            *generate_load_labelled_questions(
                BASE_DATA_PATH,
                METADATA_FOR_ANALYZER_PATH,
            ),
            # Users
            add_basic_user_statistics,
            # Learning goal analysis
            add_learning_goals_result_series(learning_goals),
            add_aggregate_learning_goal_series(learning_goals),
            add_basic_statistics_for_series("all_learning_goals_series"),
            plot_confusion_matrix(
                "execution_errors",
                "learning_goals_in_error_by_Stijn",
                "learning_goals_in_error_by_user_fix",
                True,
                group_output_dir,
            ),
            plot_confusion_matrix(
                "execution_errors",
                "learning_goals_in_error_by_Stijn",
                "learning_goals_in_error_by_error_pattern_detection",
                True,
                group_output_dir,
            ),
            plot_scatter_plot(
                "users",
                "all_learning_goals_series_average_success",
                "grade",
                group_output_dir,
            ),
            plot_scatter_plot(
                "users", "all_learning_goals_series_slope", "grade", group_output_dir
            ),
            plot_scatter_plot(
                "users",
                "all_learning_goals_series_num_practices",
                "grade",
                group_output_dir,
            ),
            plot_scatter_plot(
                "users",
                "all_learning_goals_series_num_successes",
                "grade",
                group_output_dir,
            ),
            plot_scatter_plot(
                "users", "all_learning_goals_series_num_failures", "grade", group_output_dir
            ),
            add_bayesian_knowledge_tracing(learning_goals),
            add_moving_average(learning_goals, window_size=20),
            add_basic_interaction_statistics(question_types, question_purposes),
            # Interactions part 2
            add_increase_in_success_rate,
            # overview
            add_execution_overview_df,
            add_interaction_overview_df,
            # timeline
            add_timeline_df,
            # Construct analysis
            add_construct_result_series,
            add_aggregate_construct_series,
            add_basic_statistics_for_series("all_constructs_series"),
            plot_scatter_plot(
                "users",
                "all_constructs_series_average_success",
                "grade",
                group_output_dir,
            ),
            plot_scatter_plot(
                "users", "all_constructs_series_slope", "grade", group_output_dir
            ),
            plot_scatter_plot(
                "users",
                "all_constructs_series_num_practices",
                "grade",
                group_output_dir,
            ),
            plot_scatter_plot(
                "users",
                "all_constructs_series_num_successes",
                "grade",
                group_output_dir,
            ),
            plot_scatter_plot(
                "users", "all_constructs_series_num_failures", "grade", group_output_dir
            ),
            
            # Plots: question types classification check
            plot_confusion_matrix(
                "interactions",
                "question_type_by_Thom",
                "question_type_by_ai",
                True,
                group_output_dir,
            ),
            # plot_confusion_matrix(
            #     "interactions",
            #     "question_type_by_Stijn",
            #     "question_type_by_ai",
            #     True,
            #     group_output_dir,
            # ),
            # Plots: question type vs interaction outcomes
            plot_violin_plot(
                "interactions",
                "question_type_by_ai",
                "time_until_next_edit",
                group_output_dir,
            ),
            plot_violin_plot(
                "interactions",
                "question_type_by_ai",
                "time_until_next_interaction",
                group_output_dir,
            ),
            plot_violin_plot(
                "interactions",
                "question_type_by_ai",
                "time_until_next_execution",
                group_output_dir,
            ),
            # Plots: correlation between basic interaction statistics
            plot_correlation_matrix(
                "interactions",
                [
                    "time_until_next_edit",
                    "time_until_next_interaction",
                    "time_until_next_execution",
                    "increase_in_success_rate",
                ],
                group_output_dir,
            ),
            # Plots: correlation between basic user statistics
            plot_correlation_matrix(
                "users",
                [
                    "num_interactions",
                    "num_edits",
                    "num_executions",
                    "num_executed_files",
                    "execution_success_rate",
                    "grade",
                ],
                group_output_dir,
            ),
            # Save to Excel
            write_to_excel(f"{group_output_dir}/jupyter_data.xlsx"),
        ]

        # Run the pipeline for this group
        run_pipeline(pipeline_steps)
