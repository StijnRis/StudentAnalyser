import json
import os

from loader.chatbot_log import load_chat_log
from loader.jupyter_log import load_jupyter_log
from loader.stanislas_grades import load_stanislas_grades


def generate_loader_pipeline(
    base_data_path: str, metadata_for_analyser_file: str, filter_username: str | None
):
    """Return the loader pipeline based on metadata JSON."""

    # Load metadata from JSON file
    with open(metadata_for_analyser_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Generate the pipeline
    pipeline = []

    for entry in metadata["JUPYTER_LOGS_DATA_LOCATION"]:
        path = os.path.join(base_data_path, entry["path"])
        group = entry["group"]
        pipeline.append(load_jupyter_log(path, group, filter_username))

    for entry in metadata["VOLUMES_DATA_LOCATION"]:
        path = os.path.join(base_data_path, entry["path"])
        group = entry["group"]
        pipeline.append(load_chat_log(path, group, filter_username))

    for entry in metadata["GRADES_DATA_LOCATION"]:
        path = os.path.join(base_data_path, entry["path"])
        group = entry["group"]
        pipeline.append(
            load_stanislas_grades(path, entry["max_points"], group, filter_username)
        )

    return pipeline
