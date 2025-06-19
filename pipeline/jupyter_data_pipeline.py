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
    add_time_until_next_edit,
    add_time_until_next_execution,
    add_time_until_next_interaction,
)
from loader.labelled_questions_loader import generate_load_questions_pipeline
from loader.loader_pipeline import generate_start_loader_pipeline
from messages.message_analyser import (
    add_code_in_message,
    add_included_code_snippets,
    add_message_language,
    add_message_length,
)
from pipeline.pipeline import run_pipeline
from plots.confusion_matrix import plot_confusion_matrix
from plots.scatter_plot import (
    plot_scatter_plot,
    plot_scatter_plot_with_multiple_datasets,
)
from plots.violin_plot import plot_violin_plot
from users.user_analyser import (
    add_basic_interaction_statistics,
    add_basic_learning_goals_statistics,
    add_basic_user_statistics,
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
    FILTER_USERNAMES = os.getenv("FILTER_USERNAMES", None)
    if FILTER_USERNAMES:
        FILTER_USERNAMES = FILTER_USERNAMES.split(",")
    if not METADATA_FOR_ANALYZER_PATH:
        raise ValueError(".env misses METADATA_FOR_ANALYZER_PATH")
    if not BASE_DATA_PATH:
        raise ValueError(".env misses BASE_DATA_PATH")
    if not OUTPUT_DIR:
        raise ValueError(".env misses OUTPUT_DIR")

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

    # Define your pipeline steps
    pipeline_steps: List[Callable[[Dict[str, pd.DataFrame]], None]] = [
        *generate_start_loader_pipeline(
            BASE_DATA_PATH, METADATA_FOR_ANALYZER_PATH, FILTER_USERNAMES
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
        add_time_until_next_interaction,
        add_time_until_next_edit,
        add_time_until_next_execution,
        # add_waiting_time_to_interactions,
        add_interaction_type(question_types, unknown_question_type),
        add_interaction_purpose(question_purposes),
        add_interaction_learning_goals(learning_goals),
        *generate_load_questions_pipeline(
            BASE_DATA_PATH,
            METADATA_FOR_ANALYZER_PATH,
        ),
        # Users
        add_basic_user_statistics,
        add_learning_goals_result_series(learning_goals),
        add_basic_learning_goals_statistics(learning_goals),
        add_bayesian_knowledge_tracing(learning_goals),
        add_moving_average(learning_goals, window_size=20),
        add_basic_interaction_statistics(question_types, question_purposes),
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
            "interactions",
            "question_purpose_by_question_type",
            "increase_in_success_rate",
            OUTPUT_DIR,
        ),
        plot_violin_plot(
            "interactions",
            "question_type_by_ai",
            "increase_in_success_rate",
            OUTPUT_DIR,
        ),
        plot_scatter_plot("users", "group", "num_interactions", "grade", OUTPUT_DIR),
        plot_scatter_plot("users", "group", "num_edits", "grade", OUTPUT_DIR),
        plot_scatter_plot("users", "group", "num_executions", "grade", OUTPUT_DIR),
        plot_scatter_plot("users", "group", "num_executed_files", "grade", OUTPUT_DIR),
        plot_scatter_plot(
            "users", "group", "execution_success_rate", "grade", OUTPUT_DIR
        ),
        plot_scatter_plot_with_multiple_datasets(
            "users",
            "group",
            "Question type",
            [f"num_{x.name}_questions" for x in question_types],
            "grade",
            OUTPUT_DIR,
        ),
        plot_scatter_plot_with_multiple_datasets(
            "users",
            "group",
            "Question purpose",
            [f"num_{x.name}_questions" for x in question_purposes],
            "grade",
            OUTPUT_DIR,
        ),
        plot_scatter_plot_with_multiple_datasets(
            "users",
            "group",
            "Learning goal success rate",
            [f"{x.name}_average_success" for x in learning_goals],
            "grade",
            OUTPUT_DIR,
        ),
        plot_scatter_plot_with_multiple_datasets(
            "users",
            "group",
            "Learning goal number of practices",
            [f"{x.name}_num_practices" for x in learning_goals],
            "grade",
            OUTPUT_DIR,
        ),
        plot_scatter_plot_with_multiple_datasets(
            "users",
            "group",
            "Learning goal number of successes",
            [f"{x.name}_num_successes" for x in learning_goals],
            "grade",
            OUTPUT_DIR,
        ),
        plot_scatter_plot_with_multiple_datasets(
            "users",
            "group",
            "Learning goal number of failures",
            [f"{x.name}_num_failures" for x in learning_goals],
            "grade",
            OUTPUT_DIR,
        ),
        # Plot confusion matrices for question types
        plot_confusion_matrix(
            "interactions",
            "question_type_by_Thom",
            "question_type_by_ai",
            True,
            OUTPUT_DIR,
        ),
        # plot_confusion_matrix(
        #     "interactions",
        #     "question_type_by_Stijn",
        #     "question_type_by_ai",
        #     True,
        #     OUTPUT_DIR,
        # ),
        # plot_violin_plot(
        #     "interactions",
        #     "question_type_by_ai",
        #     "time_until_next_edit",
        #     OUTPUT_DIR,
        # ),
        # plot_violin_plot(
        #     "interactions",
        #     "question_type_by_ai",
        #     "time_until_next_interaction",
        #     OUTPUT_DIR,
        # ),
        # plot_violin_plot(
        #     "interactions",
        #     "question_type_by_ai",
        #     "time_until_next_execution",
        #     OUTPUT_DIR,
        # ),
        # Save to Excel
        write_to_excel(f"{OUTPUT_DIR}/jupyter_data.xlsx"),
    ]

    # Run the pipeline
    run_pipeline(pipeline_steps)
