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
    add_error_learning_goal_by_ai_detection,
    add_error_learning_goal_by_error_pattern_detection,
    add_user_fix_analysis,
)
from executions.execution_success_analyser import (
    add_execution_successes_df,
    add_new_code_analysis,
)
from loader.jupyter_log import load_jupyter_log
from pipeline.pipeline import run_pipeline
from writer.excel import write_to_excel


def run_check_error_analyser_pipeline():

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
        add_surrounding_executions,
        # execution_success
        add_execution_successes_df,
        add_new_code_analysis(learning_goals),
        # execution_errors
        add_error_learning_goal_by_error_pattern_detection(learning_goals),
        add_user_fix_analysis(learning_goals),
        # add_error_learning_goal_by_ai_detection(learning_goals),
        # Overview
        add_execution_overview_df,
        # Save to Excel
        write_to_excel(f"{OUTPUT_DIR}/check_error_analyser.xlsx"),
    ]

    # Run the pipeline
    run_pipeline(pipeline_steps)
