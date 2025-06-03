import json
import os
from datetime import datetime
from typing import Callable, Dict

import pandas as pd


def load_chat_log(
    folder_path: str,
    group: str,
    filter_user: str | None,
) -> Callable[[Dict[str, pd.DataFrame]], None]:
    """
    Loads all chat log files in a folder and returns a pandas DataFrame with columns: time, body, sender, automated, user_id, file_path.
    """

    def load_chat_log(data: Dict[str, pd.DataFrame]) -> None:
        print(f"Loading chat logs from folder: {folder_path}")
        messages = []
        # --- USER ID LOGIC ---
        users_df = data.get(
            "users", pd.DataFrame(columns=["user_id", "group", "username"])
        )
        next_user_id = users_df["user_id"].max() + 1 if not users_df.empty else 0
        user_id_map = {
            (row["group"], row["username"]): row["user_id"]
            for _, row in users_df.iterrows()
        }
        for username in os.listdir(folder_path):
            if filter_user and username != filter_user:
                print(f"Skipping chat log from {username}")
                continue
            # Ensure user_id exists for (group, username)
            user_key = (group, username)
            if user_key not in user_id_map:
                user_id = next_user_id
                users_df = pd.concat(
                    [
                        users_df,
                        pd.DataFrame(
                            {
                                "user_id": [user_id],
                                "group": [group],
                                "username": [username],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
                user_id_map[user_key] = user_id
                next_user_id += 1
            else:
                user_id = user_id_map[user_key]
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
                            "group": group,
                            "user_id": user_id,
                            "datetime": time,
                            "body": body,
                            "sender": sender,
                            "automated": automated,
                        }
                    )
                    amount_of_messages += 1
            print(f"({amount_of_messages} messages loaded)")
        # Add data to dataframe
        messages_df = data.get("messages", pd.DataFrame())
        new_messages_df = pd.DataFrame(messages)
        messages_id_offset = (
            1 + messages_df["message_id"].max() + 1 if not messages_df.empty else 0
        )
        new_messages_df.insert(
            0, "message_id", range(messages_id_offset, messages_id_offset + len(messages))
        )
        data["messages"] = pd.concat([messages_df, new_messages_df], ignore_index=True)
        # Save updated users
        data["users"] = users_df

    return load_chat_log
