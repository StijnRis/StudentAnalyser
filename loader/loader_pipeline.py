import json
import os

from loader.chatbot_log import load_chat_log
from loader.jupyter_log import load_jupyter_log
from loader.stanislas_grades import load_stanislas_grades


def generate_start_loader_pipeline(
    base_data_path: str,
    metadata_for_analyser_file: str,
    filter_usernames: list[str] | None,
    group: str,
):
    """Return the loader pipeline based on metadata JSON. If group is specified, only load data for that group."""

    # Load metadata from JSON file
    with open(metadata_for_analyser_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Generate the pipeline
    pipeline = []

    for entry in metadata["JUPYTER_LOGS_DATA_LOCATION"]:
        if entry["group"] != group:
            continue
        path = os.path.join(base_data_path, entry["path"])
        pipeline.append(load_jupyter_log(path, filter_usernames))

    for entry in metadata["VOLUMES_DATA_LOCATION"]:
        if entry["group"] != group:
            continue
        path = os.path.join(base_data_path, entry["path"])
        pipeline.append(load_chat_log(path, filter_usernames))

    for entry in metadata["GRADES_DATA_LOCATION"]:
        if entry["group"] != group:
            continue
        path = os.path.join(base_data_path, entry["path"])
        pipeline.append(
            load_stanislas_grades(
                path, entry["max_points"], filter_usernames
            )
        )

    return pipeline
