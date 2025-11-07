import json
import os
from datetime import datetime
from typing import Callable, Dict

import pandas as pd


def load_chat_log(
    folder_path: str,
    filter_usernames: list[str] | None,
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
            row["username"]: row["user_id"]
            for _, row in users_df.iterrows()
        }
        for username in os.listdir(folder_path):
            if filter_usernames and username not in filter_usernames:
                print(f"Skipping chat log from {username}")
                continue
            # Ensure user_id exists for (group, username)
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
                    automated = msg.get("automated", msg["sender"] == "Juno")
                    messages.append(
                        {
                            "user_id": user_id,
                            "datetime": time,
                            "body": body,
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
