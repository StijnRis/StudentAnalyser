import json
import os
import re
from datetime import datetime
from typing import Callable, Dict

import pandas as pd

# --- Helper functions ---


def _parse_events_from_file(file_path: str) -> list:
    """Read and parse events from a log file, returning a list of event dicts."""
    with open(file_path, "r", encoding="utf8") as file:
        dataframe = file.read()
        if not dataframe.strip():
            return []
        return json.loads("[" + dataframe.rstrip(",\n") + "]")


def _extract_file_versions(events, username):
    """Extract file version records (code cells) from events for a given user."""
    file_versions = []
    for event_data in events:
        event_time = datetime.fromtimestamp(
            event_data["eventDetail"]["eventTime"] / 1000.0
        )

        notebook_state = event_data.get("notebookState")
        if notebook_state and "notebookContent" in notebook_state and notebook_state["notebookContent"] != None:
            file_path_val = notebook_state["notebookPath"]

            cells = notebook_state["notebookContent"]["cells"]
            for index, cell in enumerate(cells):
                if cell.get("cell_type") == "code":
                    code = cell["source"]
                    code = code.replace("\t", "")
                    file_versions.append(
                        {
                            "username": username,
                            "datetime": event_time,
                            "file": f"{file_path_val}_{index}",
                            "code": code,
                        }
                    )
    return file_versions


def _extract_executions_outputs_errors(events, username, start_execution_index):
    """Extract execution events, outputs, and errors from events for a given user."""
    executions = []
    outputs = []
    errors = []

    for event_data in events:
        event_detail = event_data["eventDetail"]
        event_type = event_detail["eventName"]
        event_time = datetime.fromtimestamp(event_detail["eventTime"] / 1000.0)

        if event_type != "CellExecuteEvent":
            continue

        notebook_state = event_data.get("notebookState")
        if notebook_state is None:
            raise ValueError(f"Notebook state is None for event: {event_data}")

        cells = notebook_state["notebookContent"]["cells"]
        if len(event_detail["eventInfo"]["cells"]) != 1:
            raise ValueError(
                f"Expected one cell in notebook content, found {len(cells)}"
            )

        executed_cell_index = event_detail["eventInfo"]["cells"][0]["index"]
        executed_cell = cells[executed_cell_index]
        file_val = notebook_state["notebookPath"]

        if executed_cell["cell_type"] != "code":
            continue

        execution_record = {
            "id": start_execution_index + len(executions),
            "username": username,
            "datetime": event_time,
            "file": f"{file_val}_{executed_cell_index}",
        }
        executions.append(execution_record)

        for output_data in executed_cell["outputs"]:
            output_type = output_data.get("output_type")
            if output_type == "error" and error_na:
                traceback = re.sub(
                    r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]",
                    "",
                    "\n".join(output_data["traceback"]),
                )
                error_record = {
                    "execution_id": execution_record["id"],
                    "error_name": output_data["ename"],
                    "error_value": output_data["evalue"],
                    "traceback": traceback,
                }
                errors.append(error_record)
            else:
                output_text = None
                if output_type == "stream":
                    output_text = output_data["text"]
                elif output_type == "execute_result":
                    output_text = output_data["data"]

                # Limit output text length for storage
                if isinstance(output_text, str) and len(output_text) > 1000:
                    output_text = output_text[:1000] + "..."
                
                output_record = {
                    "execution_id": execution_record["id"],
                    "output_type": output_type,
                    "output_text": output_text,
                }
                outputs.append(output_record)

    return executions, outputs, errors


def _extract_edits(events, username):
    """Extract edit events (cell edits, selections, etc.) from events for a given user."""
    records = []

    for event_data in events:
        event_detail = event_data["eventDetail"]
        event_type = event_detail["eventName"]
        event_time = datetime.fromtimestamp(event_detail["eventTime"] / 1000.0)
        file_val = (
            event_data["notebookState"]["notebookPath"]
            if event_data.get("notebookState")
            else None
        )
        cell_index = None
        selection = None
        file = None

        if "eventInfo" in event_detail and event_detail["eventInfo"]:
            info = event_detail["eventInfo"]

            # Add cell index
            if "index" in info:
                cell_index = info["index"]
                file = f"{file_val}_{cell_index}"
            elif (
                "cells" in info
                and info["cells"]
                and len(info["cells"]) == 1
                and "index" in info["cells"][0]
            ):
                cell_index = info["cells"][0]["index"]
                file = f"{file_val}_{cell_index}"

            # Add selection
            if "selection" in info:
                selection = info["selection"]

        records.append(
            {
                "datetime": event_time,
                "event_type": event_type,
                "file": file,
                "selection": selection,
                "username": username,
            }
        )
    return records


# --- Main loader function ---
def load_jupyter_log(
    folder_paths: list[str], filter_user: str | None = None
) -> Callable[[Dict[str, pd.DataFrame]], None]:
    """
    Loads all file version, execution, and edit logs in a folder and returns DataFrames:
    - file_versions: time, file, cell_index, username, code
    - executions: username, time, file, event_type
    - execution_success_outputs: execution_id, output_type, output_text
    - execution_errors: execution_id, error_name, error_value, traceback, output_text
    - edits: time, event_type, file, selection, username
    """

    def load_jupyter_log(
        data: Dict[str, pd.DataFrame],
    ) -> None:
        print(f"Loading Jupyter logs from folders: {folder_paths}")
        # Lists to accumulate records for each DataFrame
        file_versions = []
        executions = []
        outputs = []
        errors = []
        edits = []
        for folder_path in folder_paths:
            for file_name in os.listdir(folder_path):
                # Only process files that match the expected log file pattern
                if not (
                    file_name.startswith("jupyter-") and file_name.endswith("-log")
                ):
                    continue

                username = file_name.replace("jupyter-", "").replace("-log", "")

                # Optionally filter by user
                if filter_user and username != filter_user:
                    print(f"Skipping execution log of {username}")
                    continue

                print(f"Loading execution logs for {username}", end=" ")

                file_path = os.path.join(folder_path, file_name)
                events = _parse_events_from_file(file_path)
                print(f"(Checking {len(events)} events)")

                # Extract and accumulate all types of records
                file_versions.extend(_extract_file_versions(events, username))
                execs, outs, errs = _extract_executions_outputs_errors(
                    events, username, len(executions)
                )
                executions.extend(execs)
                outputs.extend(outs)
                errors.extend(errs)
                edits.extend(_extract_edits(events, username))

        # Convert lists to DataFrames
        file_versions_df = pd.DataFrame(file_versions)
        file_versions_df.insert(0, "id", range(len(file_versions_df)))

        executions_df = pd.DataFrame(executions)  # already has id

        successes_df = pd.DataFrame(outputs)
        successes_df.insert(0, "id", range(len(successes_df)))

        errors_df = pd.DataFrame(errors)
        errors_df.insert(0, "id", range(len(errors_df)))

        edits_df = pd.DataFrame(edits)
        edits_df.insert(0, "id", range(len(edits_df)))

        # Store DataFrames in the data dictionary
        data["file_versions"] = file_versions_df
        data["executions"] = executions_df
        data["execution_success_outputs"] = successes_df
        data["execution_errors"] = errors_df
        data["edits"] = edits_df

    return load_jupyter_log
