import os
from typing import Callable, Dict, List

import pandas as pd

from correlations.confusion_matrix import plot_confusion_matrix
from enums import get_learning_goals, get_question_purposes, get_question_types
from interactions.interaction_analyser import (
    add_interaction_overview_df,
    add_interaction_type,
)
from loader.cld_questions import load_cld_questions
from pipeline.pipeline import run_pipeline
from writer.excel import write_to_excel


def run_check_question_analyser_pipeline():
    # Environment variables
    OUTPUT_DIR = os.getenv("OUTPUT_DIR")

    # Get enums
    question_types = get_question_types()
    question_purposes = get_question_purposes()
    learning_goals = get_learning_goals()

    # Define your pipeline steps
    pipeline_steps: List[Callable[[Dict[str, pd.DataFrame]], None]] = [
        load_cld_questions(
            r"C:\University\Honours\Data\Labeled questions Thom\questions_before_midterm_labeled_by_thom_and_me.xlsx",
            question_types,
        ),
        # interactions
        add_interaction_type(question_types),
        add_interaction_overview_df,
        # Plots
        plot_confusion_matrix("interactions", "question_type_by_Thom", "question_type", OUTPUT_DIR),
        plot_confusion_matrix("interactions", "question_type_by_Stijn", "question_type", OUTPUT_DIR),
        # Save to Excel
        write_to_excel(f"{OUTPUT_DIR}/check_question_analyser.xlsx"),
    ]

    # Run the pipeline
    run_pipeline(pipeline_steps)
