import json
import os
from datetime import datetime
from typing import Callable, Dict

import pandas as pd


def load_chat_log(
    folder_paths: list[str], filter_user: str | None = None
) -> Callable[[Dict[str, pd.DataFrame]], None]:
    """
    Loads all chat log files in a folder and returns a pandas DataFrame with columns: time, body, sender, automated, username, file_path.
    """

    def load_chat_log(data: Dict[str, pd.DataFrame]) -> None:
        print(f"Loading chat logs from folders: {folder_paths}")
        messages = []
        for folder_path in folder_paths:
            for username in os.listdir(folder_path):
                if filter_user and username != filter_user:
                    print(f"Skipping chat log from {username}")
                    continue

                print(f"Loading chat logs for {username}", end=" ")

                amount_of_messages = 0
                for file_name in os.listdir(os.path.join(folder_path, username)):
                    if not file_name.endswith(".chat") or file_name.startswith(".~"):
                        continue

                    file_path = os.path.join(folder_path, username, file_name)
                    with open(file_path, "r") as file:
                        if file.read(1) == "":
                            continue
                        file.seek(0)
                        file_data = json.load(file)

                    for msg in file_data.get("messages", []):
                        time = datetime.fromtimestamp(msg["time"])
                        body = msg["body"]
                        sender = msg["sender"]
                        automated = msg.get("automated", sender == "Juno")
                        messages.append(
                            {
                                "id": len(messages) + 1,
                                "username": username,
                                "datetime": time,
                                "body": body,
                                "sender": sender,
                                "automated": automated,
                            }
                        )
                        amount_of_messages += 1
                print(f"({amount_of_messages} messages loaded)")

        dataframe = pd.DataFrame(messages)

        data["messages"] = dataframe

    return load_chat_log
