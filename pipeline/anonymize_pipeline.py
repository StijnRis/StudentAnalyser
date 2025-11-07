import os
from typing import Callable, Dict, List

import pandas as pd

from anonymization.anonymize import anonymize
from loader.loader_pipeline import generate_start_loader_pipeline
from pipeline.pipeline import run_pipeline
from writer.excel import write_to_excel


def run_anonymize_pipeline():
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
            anonymize,
            write_to_excel(f"{group_output_dir}/jupyter_data_anonymized.xlsx"),
        ]

        # Run the pipeline for this group
        run_pipeline(pipeline_steps)
