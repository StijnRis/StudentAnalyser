import json
import os
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


def _extract_file_versions(user_id, events):
    """Extract file version records (code cells) from events for a given user."""
    file_versions = []
    for event_data in events:
        event_time = datetime.fromtimestamp(
            event_data["eventDetail"]["eventTime"] / 1000.0
        )

        notebook_state = event_data.get("notebookState")
        if (
            notebook_state
            and "notebookContent" in notebook_state
            and notebook_state["notebookContent"] is not None
        ):
            file_path_val = notebook_state["notebookPath"]

            cells = notebook_state["notebookContent"]["cells"]
            for index, cell in enumerate(cells):
                if cell.get("cell_type") == "code":
                    code = cell["source"].replace("\t", "")
                    file_versions.append(
                        {
                            "user_id": user_id,
                            "datetime": event_time,
                            "filename": f"{file_path_val}_{index}",
                            "code": code,
                        }
                    )
    return file_versions


def _extract_executions_outputs_errors(user_id, events, start_execution_index):
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
            "execution_id": start_execution_index + len(executions),
            "user_id": user_id,
            "datetime": event_time,
            "filename": f"{file_val}_{executed_cell_index}",
        }
        executions.append(execution_record)
        for output_data in executed_cell["outputs"]:
            output_type = output_data.get("output_type")
            execution_id = execution_record["execution_id"]
            if output_type == "error":
                traceback = "\n".join(output_data["traceback"])
                name = output_data["ename"]
                if name == "KeyboardInterrupt":
                    output_record = {
                        "execution_id": execution_id,
                        "output_type": output_type,
                        "output_text": output_data["evalue"] + "\n" + traceback,
                    }
                    outputs.append(output_record)
                else:
                    error_record = {
                        "execution_id": execution_id,
                        "error_name": name,
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
                if isinstance(output_text, str) and len(output_text) > 1000:
                    output_text = output_text[:1000] + "..."
                output_record = {
                    "execution_id": execution_id,
                    "output_type": output_type,
                    "output_text": output_text,
                }
                outputs.append(output_record)
    return executions, outputs, errors


def _extract_edits(user_id, events):
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
            if "selection" in info:
                selection = info["selection"]
        records.append(
            {
                "user_id": user_id,
                "datetime": event_time,
                "event_type": event_type,
                "filename": file,
                "selection": selection,
            }
        )
    return records


# --- Main loader function ---
def load_jupyter_log(
    folder_path: str, filter_usernames: list[str] | None
) -> Callable[[Dict[str, pd.DataFrame]], None]:
    """
    Loads all file version, execution, and edit logs in a folder and returns DataFrames:
    - file_versions: time, file, cell_index, user_id, code
    - executions: user_id, time, file, event_type
    - execution_success_outputs: execution_id, output_type, output_text
    - execution_errors: execution_id, error_name, error_value, traceback, output_text
    - edits: time, event_type, file, selection, user_id
    """

    def load_jupyter_log(
        data: Dict[str, pd.DataFrame],
    ) -> None:
        print(f"Loading Jupyter logs from folder: {folder_path}")

        # Ensure users DataFrame exists
        users_df = data.get(
            "users", pd.DataFrame(columns=["user_id", "group", "username"])
        )
        next_user_id = users_df["user_id"].max() + 1 if not users_df.empty else 0
        user_id_map = {
            row["username"]: row["user_id"] for _, row in users_df.iterrows()
        }

        file_versions_df = data.get("file_versions", pd.DataFrame())
        executions_df = data.get("executions", pd.DataFrame())
        outputs_df = data.get("execution_success_outputs", pd.DataFrame())
        errors_df = data.get("execution_errors", pd.DataFrame())
        edits_df = data.get("edits", pd.DataFrame())

        file_version_id_offset = (
            file_versions_df["file_version_id"].max() + 1
            if not file_versions_df.empty
            else 0
        )
        execution_id_offset = (
            executions_df["execution_id"].max() + 1 if not executions_df.empty else 0
        )
        outputs_id_offset = (
            outputs_df["execution_success_output_id"].max() + 1
            if not outputs_df.empty
            else 0
        )
        errors_id_offset = (
            errors_df["execution_error_id"].max() + 1 if not errors_df.empty else 0
        )
        edits_id_offset = edits_df["edit_id"].max() + 1 if not edits_df.empty else 0

        file_versions = []
        executions = []
        outputs = []
        errors = []
        edits = []
        for file_name in os.listdir(folder_path):
            if not (file_name.startswith("jupyter-") and file_name.endswith("-log")):
                continue

            username = file_name.replace("jupyter-", "").replace("-log", "")
            # Optionally filter by user
            if filter_usernames and username not in filter_usernames:
                print(f"Skipping execution log of {username}")
                continue
            print(f"Loading execution logs for {username}", end=" ")
            file_path = os.path.join(folder_path, file_name)
            events = _parse_events_from_file(file_path)
            print(f"(Checking {len(events)} events)")

            # --- USER ID LOGIC ---
            if username not in user_id_map:
                user_id = next_user_id
                users_df = pd.concat(
                    [
                        users_df,
                        pd.DataFrame(
                            {
                                "user_id": [user_id],
                                "username": [username],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
                user_id_map[username] = user_id
                next_user_id += 1
            else:
                user_id = user_id_map[username]
            # --- Extract and accumulate all types of records, using user_id ---
            file_versions.extend(_extract_file_versions(user_id, events))
            execs, outs, errs = _extract_executions_outputs_errors(
                user_id, events, execution_id_offset + len(executions)
            )
            executions.extend(execs)
            outputs.extend(outs)
            errors.extend(errs)
            edits.extend(_extract_edits(user_id, events))

        # Save to data
        new_file_versions_df = pd.DataFrame(
            file_versions, columns=["user_id", "datetime", "filename", "code"]
        )
        new_file_versions_df.insert(
            0,
            "file_version_id",
            range(file_version_id_offset, file_version_id_offset + len(file_versions)),
        )
        data["file_versions"] = pd.concat(
            [file_versions_df, new_file_versions_df], ignore_index=True
        )

        new_executions_df = pd.DataFrame(
            executions, columns=["user_id", "datetime", "filename", "execution_id"]
        )
        data["executions"] = pd.concat(
            [executions_df, new_executions_df], ignore_index=True
        )

        new_success_outputs_df = pd.DataFrame(
            outputs, columns=["execution_id", "output_type", "output_text"]
        )
        new_success_outputs_df.insert(
            0,
            "execution_success_output_id",
            range(outputs_id_offset, outputs_id_offset + len(outputs)),
        )
        data["execution_success_outputs"] = pd.concat(
            [outputs_df, new_success_outputs_df], ignore_index=True
        )

        new_errors_df = pd.DataFrame(
            errors, columns=["execution_id", "error_name", "error_value", "traceback"]
        )
        new_errors_df.insert(
            0,
            "execution_error_id",
            range(errors_id_offset, errors_id_offset + len(errors)),
        )
        data["execution_errors"] = pd.concat(
            [errors_df, new_errors_df], ignore_index=True
        )

        new_edits_df = pd.DataFrame(
            edits,
            columns=["user_id", "datetime", "event_type", "filename", "selection"],
        )
        new_edits_df.insert(
            0, "edit_id", range(edits_id_offset, edits_id_offset + len(edits))
        )
        data["edits"] = pd.concat([edits_df, new_edits_df], ignore_index=True)

        # Save updated users
        data["users"] = users_df

    return load_jupyter_log
